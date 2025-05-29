from computers import Computer, LocalPlaywrightComputer
from utils import create_response, check_blocklisted_url
import os
import time
import json
import random
import logging
import base64
import re
from typing import List, Dict, Any, Set, Optional
from urllib.parse import urlparse

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def acknowledge_safety_check_callback(message: str) -> bool:
    response = input(f"Safety Check Warning: {message}\nDo you want to acknowledge and proceed? (y/n): ").lower()
    return response.strip() == "y"

def handle_item(item, computer: Computer, max_retries: int = 3, debug: bool = False):
    if item["type"] == "message":
        print(item["content"][0]["text"])
        return []
    if item["type"] == "computer_call":
        action = item["action"]
        action_type = action["type"]
        action_args = {k: v for k, v in action.items() if k != "type"}
        print(f"{action_type}({action_args})")
        for attempt in range(max_retries):
            try:
                getattr(computer, action_type)(**action_args)
                break
            except Exception as e:
                if attempt < max_retries - 1:
                    print(f"Attempt {attempt+1} failed: {str(e)}. Retrying...")
                    time.sleep(1)
                else:
                    print(f"Action failed after {max_retries} attempts: {str(e)}")
                    raise
        screenshot_base64 = computer.screenshot()
        if debug:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            screenshot_file = f"screenshot_{action_type}_{timestamp}.png"
            try:
                with open(screenshot_file, "wb") as f:
                    f.write(base64.b64decode(screenshot_base64))
                logger.info(f"Saved screenshot: {screenshot_file}")
            except Exception as e:
                logger.error(f"Failed to save screenshot: {str(e)}")
        pending_checks = item.get("pending_safety_checks", [])
        for check in pending_checks:
            if not acknowledge_safety_check_callback(check["message"]):
                raise ValueError(f"Safety check failed: {check['message']}")
        call_output = {
            "type": "computer_call_output",
            "call_id": item["call_id"],
            "acknowledged_safety_checks": pending_checks,
            "output": {
                "type": "input_image",
                "image_url": f"data:image/png;base64,{screenshot_base64}",
            },
        }
        if computer.environment == "browser":
            try:
                current_url = computer.get_current_url()
                call_output["output"]["current_url"] = current_url
                check_blocklisted_url(current_url)
            except Exception as e:
                print(f"Warning: Could not get current URL: {str(e)}")
        return [call_output]
    return []

def send_message_to_agent(message: str, items: list, computer: Computer, tools: list, max_attempts: int = 3, debug: bool = False) -> list:
    cleaned_items = []
    skip_next = False
    for item in items:
        if skip_next:
            skip_next = False
            continue
        if item.get("type") == "reasoning":
            logger.warning(f"Found reasoning item: {item}. Ensuring follow-up message.")
            cleaned_items.append(item)
            cleaned_items.append({"role": "user", "content": [{"type": "input_text", "text": "Please continue with the response."}]})
            skip_next = True
        else:
            cleaned_items.append(item)
    
    cleaned_items.append({"role": "user", "content": [{"type": "input_text", "text": message}]})

    attempt = 0
    while attempt < max_attempts:
        try:
            if debug:
                logger.debug(f"Input items for attempt {attempt+1}: {cleaned_items}")
            
            response = create_response(model="computer-use-preview", input=cleaned_items, tools=tools, truncation="auto")
            
            if "output" not in response:
                logger.warning(f"No output from model. Response: {response}")
                attempt += 1
                if attempt >= max_attempts:
                    raise ValueError("No output from model after multiple attempts")
                time.sleep(2)
                continue
            
            for item in response["output"]:
                if "content" in item:
                    content = item["content"]
                    if isinstance(content, list):
                        for content_item in content:
                            if content_item.get("type") == "text":
                                content_item["type"] = "output_text"
                    elif isinstance(content, str):
                        item["content"] = [{"type": "output_text", "text": content}]
            
            cleaned_items += response["output"]
            
            for item in response["output"]:
                try:
                    new_items = handle_item(item, computer, debug=debug)
                    if new_items:
                        cleaned_items += new_items
                except Exception as e:
                    logger.error(f"Error handling item: {str(e)}")
                    error_message = f"There was an error performing the last action: {str(e)}. Please try a different approach."
                    cleaned_items.append({"role": "user", "content": [{"type": "input_text", "text": error_message}]})
            
            if cleaned_items[-1].get("role") == "assistant":
                break
            
            if len(response["output"]) == 0:
                logger.warning("Empty response from agent. Retrying...")
                attempt += 1
                if attempt >= max_attempts:
                    raise ValueError("Empty responses from agent after multiple attempts")
                time.sleep(2)
                
        except Exception as e:
            logger.error(f"Error in agent communication: {str(e)}")
            if debug:
                logger.debug(f"Items at error: {cleaned_items}")
            attempt += 1
            if attempt >= max_attempts:
                logger.error("Failed to communicate with agent after multiple attempts")
                cleaned_items.append({"role": "system", "content": [{"type": "input_text", "text": f"Communication error: {str(e)}"}]})
                break
            time.sleep(2)
    
    return cleaned_items

class WebsiteTestGenerator:
    def __init__(self, computer: Computer, base_url: str, max_flows: int = 10, output_file: str = "test_flows.txt", retry_attempts: int = 3, wait_time: float = 0.5, debug: bool = False, priority_category: Optional[str] = None):
        self.computer = computer
        self.base_url = base_url
        self.max_flows = max_flows
        self.output_file = output_file
        self.flows: List[Dict[str, Any]] = []
        self.visited_pages: Set[str] = set()
        self.retry_attempts = retry_attempts
        self.wait_time = wait_time
        self.debug = debug
        self.priority_category = priority_category
        display_width, display_height = computer.dimensions
        self.tools = [{"type": "computer-preview", "display_width": display_width, "display_height": display_height, "environment": computer.environment}]
        self.priority_keywords = {
            "high": ["checkout", "payment", "cart", "buy", "purchase", "login", "signin", "signup", "register", "account"],
            "medium": ["product", "search", "filter", "category", "collection", "wishlist", "favorite"],
            "low": ["about", "contact", "faq", "help", "support", "policy", "newsletter"]
        }
        self.user_personas = [
            {"name": "New Visitor", "behavior": "browsing", "patience": "low"},
            {"name": "Returning Customer", "behavior": "purposeful", "patience": "medium"},
            {"name": "Power User", "behavior": "efficient", "patience": "high"}
        ]

    def navigate_to_url(self, url: str, max_attempts: int = 3) -> bool:
        for attempt in range(max_attempts):
            try:
                logger.info(f"Navigating to: {url} (Attempt {attempt+1}/{max_attempts})")
                if hasattr(self.computer, "navigate_to"):
                    self.computer.navigate_to(url)
                elif hasattr(self.computer, "goto"):
                    self.computer.goto(url)
                elif hasattr(self.computer, "open"):
                    self.computer.open(url)
                else:
                    navigation_items = []
                    navigation_prompt = f"Navigate to this URL: {url}. If the page doesn't load properly, refresh it once."
                    send_message_to_agent(navigation_prompt, navigation_items, self.computer, self.tools, debug=self.debug)
                time.sleep(3)
                current_url = self.computer.get_current_url()
                logger.info(f"Navigated to: {current_url}")
                target_domain = urlparse(url).netloc
                current_domain = urlparse(current_url).netloc
                return target_domain in current_domain or current_domain in target_domain
            except Exception as e:
                logger.error(f"Error navigating to {url} (Attempt {attempt+1}): {e}")
                if attempt < max_attempts - 1:
                    time.sleep(3)
                else:
                    return False
        return False

    def normalize_url(self, url: str) -> str:
        parsed = urlparse(url)
        return f"{parsed.scheme}://{parsed.netloc}{parsed.path}"

    def determine_priority(self, flow_data: Dict[str, Any]) -> str:
        url = flow_data.get("ending_url", "").lower()
        name = flow_data.get("name", "").lower()
        description = flow_data.get("description", "").lower()
        if self.priority_category and self.priority_category.lower() in (url + name + description):
            return "high"
        combined_text = f"{url} {name} {description}"
        for priority, keywords in self.priority_keywords.items():
            if any(keyword in combined_text for keyword in keywords):
                return priority
        return "medium"

    def discover_website_structure(self) -> List[Dict[str, Any]]:
        logger.info(f"Analyzing website structure: {self.base_url}")
        success = self.navigate_to_url(self.base_url)
        if not success:
            logger.warning("Initial navigation failed. Retrying with basic prompt.")
            navigation_items = []
            navigation_prompt = "The website navigation may have failed. Look at what's currently on screen and identify any main functionality you can see."
            send_message_to_agent(navigation_prompt, navigation_items, self.computer, self.tools, debug=self.debug)
        priority_prompt = f"Focus on flows related to {self.priority_category} if possible. " if self.priority_category else ""
        prompt = f"""
        You are a website testing expert analyzing {self.base_url}.
        Explore this website for 30-60 seconds to understand its structure.
        Look at navigation menus, footer links, and main content areas.
        Identify the most important user journeys to test, focusing on:
        1. Critical paths (checkout, login, signup)
        2. Core functionality (search, filtering, forms)
        3. Common user tasks (browsing, comparing, saving)
        {priority_prompt}For each flow, return a JSON object with:
        - name: Descriptive name of the flow
        - description: Why users perform this flow
        - priority: High, Medium, or Low
        - steps: List of steps, each with a description (e.g., "Click the 'Login' button")
        Return a JSON array of flow objects. Example:
        [
            {{
                "name": "User Journey: Login",
                "description": "Log in to access account",
                "priority": "High",
                "steps": [
                    "Click the 'Login/Signup' button",
                    "Enter '+919876543210' in the mobile number field",
                    "Enter '123456' in the OTP field"
                ]
            }}
        ]
        Ensure the response is valid JSON. Do not include extra text or Markdown outside the JSON array.
        If no flows are identified, return an empty array [].
        """
        discovery_items = send_message_to_agent(prompt, [], self.computer, self.tools, debug=self.debug)
        if not discovery_items:
            logger.warning("Failed to get website analysis. Using fallback approach.")
            return self.generate_fallback_flows()
        flows_data = self.extract_flows_from_response(discovery_items)
        if not flows_data or len(flows_data) < 2:
            logger.warning("Insufficient flows extracted. Using enhanced analysis approach.")
            return self.generate_enhanced_flows()
        return flows_data

    def generate_fallback_flows(self) -> List[Dict[str, Any]]:
        logger.warning("Using fallback flow generation due to discovery failure.")
        self.navigate_to_url(self.base_url)
        priority_prompt = f"Focus on flows related to {self.priority_category} if possible. " if self.priority_category else ""
        fallback_prompt = f"""
        The website analysis didn't work as expected.
        Look at the current page of {self.base_url} and identify at least 3 basic actions a user can take.
        For each action, create a test flow as a JSON object with:
        - name: Flow name
        - description: Brief description
        - priority: High, Medium, or Low
        - steps: List of 2-5 step descriptions
        {priority_prompt}Return a JSON array of flow objects. Example:
        [
            {{
                "name": "Basic Navigation",
                "description": "Navigate the homepage",
                "priority": "Medium",
                "steps": [
                    "Visit the homepage",
                    "Click a visible menu link"
                ]
            }}
        ]
        Ensure the response is valid JSON. Do not include extra text or Markdown outside the JSON array.
        """
        fallback_items = send_message_to_agent(fallback_prompt, [], self.computer, self.tools, debug=self.debug)
        flows_data = self.extract_flows_from_response(fallback_items)
        if not flows_data or len(flows_data) < 1:
            logger.warning("Creating default test flows as last resort")
            try:
                screenshot_base64 = self.computer.screenshot()
                screenshot_prompt = f"""
                Look at this screenshot of the website.
                Identify 2-3 obvious interactive elements.
                For each, create a test flow as a JSON object with:
                - name: Flow name
                - description: Brief description
                - priority: High, Medium, or Low
                - steps: List of step descriptions
                {priority_prompt}Return a JSON array of flow objects.
                Ensure the response is valid JSON. Do not include extra text or Markdown outside the JSON array.
                """
                screenshot_items = [{"role": "user", "content": [{"type": "input_text", "text": screenshot_prompt}, {"type": "input_image", "image_url": {"url": f"data:image/png;base64,{screenshot_base64}"}}]}]
                screenshot_items = send_message_to_agent(screenshot_prompt, screenshot_items, self.computer, self.tools, debug=self.debug)
                flows_data = self.extract_flows_from_response(screenshot_items)
            except Exception as e:
                logger.error(f"Screenshot-based flow generation failed: {str(e)}")
            if not flows_data or len(flows_data) < 1:
                flows_data = [
                    {
                        "name": "Basic Navigation",
                        "description": "Test basic navigation",
                        "priority": "high",
                        "steps": ["Visit the homepage", "Wait for 2 seconds", "Click any visible link"]
                    },
                    {
                        "name": "Page Interaction",
                        "description": "Test basic page interaction",
                        "priority": "medium",
                        "steps": ["Visit the homepage", "Scroll down the page"]
                    }
                ]
        return flows_data

    def generate_enhanced_flows(self) -> List[Dict[str, Any]]:
        logger.info("Using enhanced flow generation with user personas...")
        self.navigate_to_url(self.base_url)
        persona = random.choice(self.user_personas)
        priority_prompt = f"Focus on flows related to {self.priority_category} if possible. " if self.priority_category else ""
        enhanced_prompt = f"""
        User Persona: {persona['name']}
        Behavior: {persona['behavior']}
        Patience: {persona['patience']}
        As this user on {self.base_url}, create 3 test flows as JSON objects:
        - name: Descriptive flow name
        - description: User goal
        - priority: High, Medium, or Low
        - steps: List of specific step descriptions (e.g., "Click the 'Login' button")
        {priority_prompt}Return a JSON array of flow objects. Example:
        [
            {{
                "name": "User Journey: Login",
                "description": "Log in to access account",
                "priority": "High",
                "steps": [
                    "Click the 'Login/Signup' button",
                    "Enter '+919876543210' in the mobile number field",
                    "Enter '123456' in the OTP field"
                ]
            }}
        ]
        Ensure the response is valid JSON. Do not include extra text or Markdown outside the JSON array.
        """
        enhanced_items = send_message_to_agent(enhanced_prompt, [], self.computer, self.tools, debug=self.debug)
        flows_data = self.extract_flows_from_response(enhanced_items)
        for flow in flows_data:
            flow["user_persona"] = persona["name"]
        return flows_data

    def extract_flows_from_response(self, items: list) -> List[Dict[str, Any]]:
        flows_data = []
        assistant_message = ""
        for item in reversed(items):
            if item.get("role") == "assistant" and "content" in item:
                content = item["content"]
                if isinstance(content, list):
                    for content_item in content:
                        if content_item.get("type") == "output_text":
                            assistant_message = content_item.get("text", "")
                            break
                elif isinstance(content, str):
                    assistant_message = content
                if assistant_message:
                    break
        if not assistant_message:
            logger.warning("No assistant response found for flow extraction")
            if self.debug:
                logger.debug(f"Raw items: {items}")
            return flows_data
        if self.debug:
            with open(f"assistant_response_{time.strftime('%Y%m%d_%H%M%S')}.txt", "w") as f:
                f.write(assistant_message)
            logger.info("Saved assistant response for debugging")
        try:
            flows_data = json.loads(assistant_message)
            if not isinstance(flows_data, list):
                logger.warning("Agent response is not a JSON array")
                return []
        except json.JSONDecodeError:
            logger.error("Failed to parse agent response as JSON")
            return []
        return self._validate_flows(flows_data)

    def _validate_flows(self, flows_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        validated_flows = []
        for flow in flows_data:
            if not isinstance(flow, dict):
                logger.warning(f"Invalid flow format: {flow}. Skipping.")
                continue
            if "name" not in flow or not flow["name"]:
                logger.warning(f"Flow missing name: {flow}. Skipping.")
                continue
            if "description" not in flow or not flow["description"]:
                logger.warning(f"Flow '{flow['name']}' missing description. Assigning default.")
                flow["description"] = "No description provided"
            if "priority" not in flow or not flow["priority"] or flow["priority"].lower() not in ["high", "medium", "low"]:
                logger.info(f"Flow '{flow['name']}' missing or invalid priority. Determining automatically.")
                flow["priority"] = self.determine_priority(flow)
            if "steps" not in flow or not flow["steps"] or not isinstance(flow["steps"], list) or not all(isinstance(step, str) for step in flow["steps"]):
                logger.warning(f"Flow '{flow['name']}' missing or invalid steps. Assigning default.")
                flow["steps"] = [f"Visit {self.base_url}"]
            validated_flows.append(flow)
        if not validated_flows:
            logger.warning("No flows validated. Creating default flow.")
            validated_flows.append({
                "name": "Basic Site Navigation",
                "description": "Navigate through the main sections",
                "priority": "high",
                "steps": [f"Visit {self.base_url}", "Wait for 2 seconds", "Click any visible link"]
            })
        logger.info(f"Validated {len(validated_flows)} flows")
        return validated_flows

    def execute_flow(self, flow: Dict[str, Any]) -> Dict[str, Any]:
        logger.info(f"Executing flow: {flow['name']} (Priority: {flow['priority']})")
        result = {
            "flow_name": flow["name"],
            "success": True,
            "steps_completed": 0,
            "total_steps": len(flow["steps"]),
            "errors_encountered": [],
            "errors_resolved": [],
            "ending_url": None,
            "duration_seconds": 0
        }
        start_time = time.time()
        reset_success = self.navigate_to_url(self.base_url)
        if not reset_success:
            result["success"] = False
            result["errors_encountered"].append("Failed to navigate to base URL")
            return result
        for i, step in enumerate(flow["steps"]):
            try:
                logger.info(f"Step {i+1}/{len(flow['steps'])}: {step}")
                time.sleep(self.wait_time + random.uniform(0.2, 1.0))
                step_prompt = f"""
                On the website {self.base_url}, perform the following action:
                - Step: {step}
                Current URL: {self.computer.get_current_url()}
                If the action involves clicking, find the relevant button or link.
                If it involves typing, locate the appropriate input field.
                If it involves verification, check for the specified content.
                If the action is unclear, suggest the best way to proceed.
                """
                step_items = send_message_to_agent(step_prompt, [], self.computer, self.tools, debug=self.debug)
                verification_failed = any("failed" in item.get("content", [{}])[0].get("text", "").lower() for item in step_items if item.get("role") == "assistant")
                if verification_failed:
                    error_msg = f"Step failed: {step}"
                    result["errors_encountered"].append(error_msg)
                    recover_prompt = f"Action '{step}' failed. Suggest an alternative approach."
                    send_message_to_agent(recover_prompt, [], self.computer, self.tools, debug=self.debug)
                    result["errors_resolved"].append(f"Attempted recovery for: {error_msg}")
                result["steps_completed"] += 1
            except Exception as e:
                error_msg = f"Error at step {i+1}: {str(e)}"
                result["errors_encountered"].append(error_msg)
                if i < len(flow["steps"]) - 1:
                    try:
                        recover_prompt = f"Error occurred: {str(e)}. Suggest an alternative approach for '{step}'."
                        send_message_to_agent(recover_prompt, [], self.computer, self.tools, debug=self.debug)
                        result["errors_resolved"].append(f"Attempted recovery for: {error_msg}")
                        continue
                    except Exception as recover_e:
                        logger.error(f"Recovery failed: {str(recover_e)}")
                result["success"] = False
                break
        try:
            result["ending_url"] = self.computer.get_current_url()
        except Exception as e:
            logger.error(f"Could not get ending URL: {str(e)}")
        result["duration_seconds"] = round(time.time() - start_time, 2)
        if result["errors_encountered"] and not result["errors_resolved"]:
            result["success"] = False
        status = "Passed" if result["success"] else f"Failed at step {result['steps_completed'] + 1}"
        logger.info(f"{status} - {result['steps_completed']}/{result['total_steps']} steps completed in {result['duration_seconds']}s")
        self.get_agent_test_assessment(flow, result)
        return result

    def get_agent_test_assessment(self, flow: Dict[str, Any], result: Dict[str, Any]) -> None:
        if not result["success"] or result["errors_encountered"]:
            try:
                assessment_prompt = f"""
                Review this test flow:
                Test Flow: {flow['name']}
                Steps Completed: {result['steps_completed']} of {result['total_steps']}
                Errors: {result['errors_encountered']}
                Recovery Attempts: {result['errors_resolved']}
                Analyze:
                1. Likely cause of error(s)
                2. Suggestions to improve the test
                3. Alternative path for the same goal
                Keep it brief and practical.
                """
                assessment_items = send_message_to_agent(assessment_prompt, [], self.computer, self.tools, debug=self.debug)
                assessment = None
                for item in assessment_items[::-1]:
                    if item.get("role") == "assistant" and "content" in item:
                        content = item["content"]
                        if isinstance(content, list):
                            for content_item in content:
                                if content_item.get("type") == "output_text":
                                    assessment = content_item.get("text", "")
                                    break
                        if assessment:
                            break
                if assessment:
                    logger.info("Agent Assessment:")
                    logger.info(assessment.strip().replace('\n', '\n  '))
                    result["agent_assessment"] = assessment
            except Exception as e:
                logger.error(f"Could not get agent assessment: {str(e)}")

    def save_flows_to_text(self, flows: List[Dict[str, Any]], output_file: str) -> None:
        try:
            with open(output_file, "w") as f:
                for flow in flows:
                    f.write(f"Flow: {flow['name']}\n")
                    f.write(f"Description: {flow['description']}\n")
                    f.write(f"Priority: {flow['priority']}\n")
                    if "user_persona" in flow:
                        f.write(f"User Persona: {flow['user_persona']}\n")
                    f.write("Steps:\n")
                    for i, step in enumerate(flow['steps'], 1):
                        f.write(f"  {i}. {step}\n")
                    f.write("\n")
            logger.info(f"Saved {len(flows)} test flows to {output_file}")
        except Exception as e:
            logger.error(f"Error saving flows to {output_file}: {str(e)}")

    def read_flows_from_text(self, input_file: str) -> List[Dict[str, Any]]:
        flows = []
        current_flow = None
        try:
            with open(input_file, "r") as f:
                lines = f.readlines()
                for line in lines:
                    line = line.strip()
                    if not line:
                        continue
                    if line.startswith("Flow:"):
                        if current_flow:
                            flows.append(current_flow)
                        current_flow = {"name": line[5:].strip(), "steps": []}
                    elif line.startswith("Description:") and current_flow:
                        current_flow["description"] = line[12:].strip()
                    elif line.startswith("Priority:") and current_flow:
                        current_flow["priority"] = line[9:].strip()
                    elif line.startswith("User Persona:") and current_flow:
                        current_flow["user_persona"] = line[13:].strip()
                    elif line.startswith("Steps:"):
                        continue
                    elif current_flow and re.match(r"\s*\d+\.", line):
                        step_match = re.match(r"\s*\d+\.\s*(.+)", line)
                        if step_match:
                            current_flow["steps"].append(step_match.group(1).strip())
                if current_flow:
                    flows.append(current_flow)
            logger.info(f"Read {len(flows)} flows from {input_file}")
            return self._validate_flows(flows)
        except Exception as e:
            logger.error(f"Error reading flows from {input_file}: {str(e)}")
            return []

    def generate_and_execute_test_flows(self) -> Dict[str, Any]:
        results = {
            "website": self.base_url,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "flows": [],
            "test_results": [],
            "summary": {"total_flows": 0, "successful_flows": 0, "failed_flows": 0, "errors_encountered": 0, "errors_resolved": 0, "total_duration_seconds": 0}
        }
        try:
            logger.info(f"Starting analysis of {self.base_url}")
            flows_data = self.discover_website_structure()
            if not flows_data:
                flows_data = self.generate_fallback_flows()
            flows_data = flows_data[:self.max_flows]
            priority_rank = {"high": 0, "medium": 1, "low": 2}
            flows_data.sort(key=lambda x: priority_rank.get(x.get("priority", "medium").lower(), 1))
            results["flows"] = flows_data
            self.save_flows_to_text(flows_data, self.output_file)
            execute = input("\n🔍 Do you want to execute the test flows? (y/n): ").lower().strip()
            if execute != "y":
                return results
            customize = input("Do you want to customize test execution parameters? (y/n): ").lower().strip()
            if customize == "y":
                try:
                    num_flows = input(f"Number of flows to execute (default: {len(flows_data)}): ")
                    if num_flows.strip():
                        flows_data = flows_data[:int(num_flows)]
                    self.wait_time = float(input(f"Wait time between actions in seconds (default: {self.wait_time}): ") or self.wait_time)
                except ValueError:
                    logger.error("Invalid input. Using default values.")
            start_time = time.time()
            flows_to_execute = self.read_flows_from_text(self.output_file)
            for i, flow in enumerate(flows_to_execute):
                logger.info(f"[{i+1}/{len(flows_to_execute)}] Testing flow: {flow['name']} (Priority: {flow['priority']})")
                try:
                    result = self.execute_flow(flow)
                    results["test_results"].append(result)
                    if result["success"]:
                        results["summary"]["successful_flows"] += 1
                    else:
                        results["summary"]["failed_flows"] += 1
                    results["summary"]["errors_encountered"] += len(result.get("errors_encountered", []))
                    results["summary"]["errors_resolved"] += len(result.get("errors_resolved", []))
                    self.save_results(results, "test_results_interim.txt")
                except Exception as e:
                    logger.error(f"Critical error executing flow {flow['name']}: {str(e)}")
                    results["test_results"].append({"flow_name": flow["name"], "success": False, "critical_error": str(e), "steps_completed": 0, "total_steps": len(flow["steps"]), "errors_encountered": [str(e)], "errors_resolved": []})
                    results["summary"]["failed_flows"] += 1
                time.sleep(2)
            results["summary"]["total_duration_seconds"] = round(time.time() - start_time, 2)
            results["summary"]["total_flows"] = len(results["test_results"])
            self.print_test_summary(results)
            self.generate_overall_assessment(results)
            self.save_final_report(results)
        except Exception as e:
            logger.error(f"Critical error in test generation process: {str(e)}")
            results["critical_error"] = str(e)
        self.save_results(results)
        return results

    def print_test_summary(self, results: Dict[str, Any]) -> None:
        summary = results["summary"]
        print("\n" + "="*60)
        print(f" TEST SUMMARY FOR {results['website']}")
        print("="*60)
        print(f"Total Flows: {summary['total_flows']}")
        print(f" Successful: {summary['successful_flows']} ({round(summary['successful_flows']/max(1, summary['total_flows'])*100)}%)")
        print(f"Failed: {summary['failed_flows']}")
        print(f"⚠️ Errors Encountered: {summary['errors_encountered']}")
        print(f" Errors Resolved: {summary['errors_resolved']}")
        print(f" Total Duration: {summary['total_duration_seconds']} seconds")
        print("="*60)
        print("\nFLOW RESULTS:")
        for result in results["test_results"]:
            status = "Pass" if result["success"] else " Fail"
            print(f"{status} - {result['flow_name']} ({result['steps_completed']}/{result['total_steps']} steps)")

    def generate_overall_assessment(self, results: Dict[str, Any]) -> None:
        try:
            self.navigate_to_url(self.base_url)
            flow_results = [f"- {result['flow_name']}: {'Passed' if result['success'] else 'Failed'} ({result['steps_completed']}/{result['total_steps']} steps)" for result in results["test_results"]]
            flow_results_text = "\n".join(flow_results)
            assessment_prompt = f"""
            Based on testing {self.base_url}, assess:
            1. Overall website usability
            2. Critical issues discovered
            3. Key recommendations for improvement
            Test results:
            {flow_results_text}
            Success rate: {results["summary"]["successful_flows"]}/{results["summary"]["total_flows"]} flows
            Errors: {results["summary"]["errors_encountered"]}
            Resolved: {results["summary"]["errors_resolved"]}
            Provide a professional, balanced assessment.
            """
            assessment_items = send_message_to_agent(assessment_prompt, [], self.computer, self.tools, debug=self.debug)
            assessment = None
            for item in assessment_items[::-1]:
                if item.get("role") == "assistant" and "content" in item:
                    content = item["content"]
                    if isinstance(content, list):
                        for content_item in content:
                            if content_item.get("type") == "output_text":
                                assessment = content_item.get("text", "")
                                break
                    if assessment:
                        break
            if assessment:
                results["overall_assessment"] = assessment
                print("\n OVERALL ASSESSMENT:")
                print("="*60)
                print(assessment)
                print("="*60)
        except Exception as e:
            logger.error(f"Could not generate overall assessment: {str(e)}")

    def save_results(self, results: Dict[str, Any], output_file: str = None) -> None:
        if output_file is None:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            domain = urlparse(self.base_url).netloc.replace(".", "_")
            output_file = f"test_results_{domain}_{timestamp}.txt"
        try:
            with open(output_file, "w") as f:
                f.write(f"Website: {results['website']}\n")
                f.write(f"Timestamp: {results['timestamp']}\n\n")
                f.write("Test Results:\n")
                for result in results["test_results"]:
                    f.write(f"Flow: {result['flow_name']}\n")
                    f.write(f"Status: {'Passed' if result['success'] else 'Failed'}\n")
                    f.write(f"Steps Completed: {result['steps_completed']}/{result['total_steps']}\n")
                    f.write(f"Duration: {result['duration_seconds']} seconds\n")
                    if result.get("errors_encountered"):
                        f.write(f"Errors: {', '.join(result['errors_encountered'])}\n")
                    if result.get("errors_resolved"):
                        f.write(f"Resolved: {', '.join(result['errors_resolved'])}\n")
                    if result.get("agent_assessment"):
                        f.write(f"Assessment: {result['agent_assessment']}\n")
                    f.write("\n")
                f.write("Summary:\n")
                f.write(f"Total Flows: {results['summary']['total_flows']}\n")
                f.write(f"Successful: {results['summary']['successful_flows']}\n")
                f.write(f"Failed: {results['summary']['failed_flows']}\n")
                f.write(f"Errors Encountered: {results['summary']['errors_encountered']}\n")
                f.write(f"Errors Resolved: {results['summary']['errors_resolved']}\n")
                f.write(f"Total Duration: {results['summary']['total_duration_seconds']} seconds\n")
                if results.get("overall_assessment"):
                    f.write(f"\nOverall Assessment:\n{results['overall_assessment']}\n")
            logger.info(f"Results saved to {output_file}")
        except Exception as e:
            logger.error(f"Error saving results to {output_file}: {str(e)}")

    def save_final_report(self, results: Dict[str, Any]) -> None:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        domain = urlparse(self.base_url).netloc.replace(".", "_")
        report_file = f"final_test_report_{domain}_{timestamp}.txt"
        try:
            with open(report_file, "w") as f:
                f.write(f"Final Test Report for {results['website']}\n")
                f.write(f"Generated on: {results['timestamp']}\n")
                f.write("="*60 + "\n")
                f.write(f"Total Flows Tested: {results['summary']['total_flows']}\n")
                f.write(f"Successful Flows: {results['summary']['successful_flows']} ({round(results['summary']['successful_flows']/max(1, results['summary']['total_flows'])*100)}%)\n")
                f.write(f"Failed Flows: {results['summary']['failed_flows']}\n")
                f.write(f"Errors Encountered: {results['summary']['errors_encountered']}\n")
                f.write(f"Errors Resolved: {results['summary']['errors_resolved']}\n")
                f.write(f"Total Duration: {results['summary']['total_duration_seconds']} seconds\n")
                f.write("\nDetailed Results:\n")
                for result in results["test_results"]:
                    f.write(f"Flow: {result['flow_name']}\n")
                    f.write(f"Status: {'Passed' if result['success'] else 'Failed'}\n")
                    f.write(f"Steps: {result['steps_completed']}/{result['total_steps']}\n")
                    if result.get("errors_encountered"):
                        f.write(f"Errors: {', '.join(result['errors_encountered'])}\n")
                    if result.get("agent_assessment"):
                        f.write(f"Agent Assessment: {result['agent_assessment']}\n")
                    f.write("\n")
                if results.get("overall_assessment"):
                    f.write(f"Overall Assessment:\n{results['overall_assessment']}\n")
                f.write("="*60 + "\n")
                f.write(f"Report saved to: {report_file}\n")
            logger.info(f"Final report saved to {report_file}")
        except Exception as e:
            logger.error(f"Error saving final report to {report_file}: {str(e)}")

def main():
    print("=" * 75)
    website_url = input("Enter the website URL to test (e.g., https://example.com): ")
    if not website_url.startswith(("http://", "https://")):
        website_url = "https://" + website_url
    try:
        max_flows = int(input("Maximum number of test flows (default: 5): ") or "5")
    except ValueError:
        max_flows = 5
    output_file = input("Output text file name (default: test_flows.txt): ") or "test_flows.txt"
    try:
        wait_time = float(input("Wait time between actions in seconds (default: 0.5): ") or "0.5")
    except ValueError:
        wait_time = 0.5
    headless_mode = input("Run in headless mode? (y/n, default: n): ").lower().strip() == "y"
    debug_mode = input("Enable debug mode? (y/n, default: n): ").lower().strip() == "y"
    priority_category = input("Enter a flow category to prioritize (optional): ").strip() or None
    print("\nStarting browser...\n")
    try:
        with LocalPlaywrightComputer(headless=headless_mode) as computer:
            test_generator = WebsiteTestGenerator(computer=computer, base_url=website_url, max_flows=max_flows, output_file=output_file, wait_time=wait_time, debug=debug_mode, priority_category=priority_category)
            test_generator.generate_and_execute_test_flows()
            print("\nTest generation and execution complete!")
    except Exception as e:
        logger.error(f"Critical error: {str(e)}")
        print("\nPlease check your installation and network connection, then try again.")

if __name__ == "__main__":
    main()