import streamlit as st
import google.generativeai as genai
from PIL import Image
import io
import json
import numpy as np
from ultralytics.utils.plotting import Annotator, colors
from typing import List, Dict, Any, Optional
from google.generativeai import types
import traceback

class DetectionProcessor:
    """
    Handles object detection tasks using the Gemini model, including processing
    the model's response, converting annotations to different formats, and
    visualizing the results on images.
    """

    def __init__(self, model: genai.GenerativeModel, temperature: float) -> None:
        """
        Initializes the DetectionProcessor.

        Args:
            model (genai.GenerativeModel): The configured GenerativeModel instance from google.generativeai.
            temperature (float): The temperature setting for the Gemini model's generation.
        """
        self.model = model
        self.temperature = temperature

    def _clean_results(self, results: str) -> str:
        """
        Removes potential markdown formatting (like ```json ... ```) from the
        raw string response to isolate the JSON content.

        Args:
            results (str): The raw string response from the Gemini model.

        Returns:
            str: The cleaned string ready for JSON parsing.
        """
        cleaned = results.strip()
        if cleaned.startswith("```json"):
            cleaned = cleaned[7:]
        if cleaned.endswith("```"):
            cleaned = cleaned[:-3]
        return cleaned.strip()

    def process_detection_results(self, results: str, image: np.ndarray, format_choice: str) -> Optional[List[Dict[str, Any]]]:
        """
        Parses the raw detection results string from Gemini, validates it,
        and converts the annotations into the specified format (JSON, YOLO, COCO).

        Args:
            results (str): The raw string response from the Gemini model, expected
                     to contain JSON-like detection data.
            image (np.ndarray): The input image as a NumPy array (used for height/width).
            format_choice (str): The desired output format ('JSON', 'YOLO', 'COCO').

        Returns:
            Optional[List[Dict[str, Any]]]: The processed annotations in the specified format (list for JSON/YOLO,
            dict for COCO).
        """
        try:
            cleaned_results_str = self._clean_results(results)
            cln_results = None
            try:
                cln_results = json.loads(cleaned_results_str)
            except json.JSONDecodeError:
                try:
                    corrected_str = cleaned_results_str.replace("'", '"')
                    cln_results = json.loads(corrected_str)
                except json.JSONDecodeError:
                    try:
                        start = cleaned_results_str.find('[')
                        end = cleaned_results_str.rfind(']') + 1
                        if start != -1 and end != 0:
                            json_str_attempt = cleaned_results_str[start:end]
                            cln_results = json.loads(json_str_attempt.replace("'", '"'))
                        else:
                            start = cleaned_results_str.find('{')
                            end = cleaned_results_str.rfind('}') + 1
                            if start != -1 and end != 0:
                                json_str_attempt = cleaned_results_str[start:end]
                                cln_results = json.loads(json_str_attempt.replace("'", '"'))
                            else:
                                raise json.JSONDecodeError("Could not find JSON array or object", cleaned_results_str, 0)
                    except json.JSONDecodeError as final_json_e:
                         st.error(f"Could not extract valid JSON even after attempting cleanup: {final_json_e}")
                         st.text_area("Cleaned Gemini Response (Failed JSON):", value=cleaned_results_str, height=150)
                         st.text_area("Raw Gemini Response:", value=results, height=150)
                         return None

            if cln_results is None:
                 st.error("Failed to obtain valid JSON data after all parsing attempts.")
                 return None

            if not isinstance(cln_results, list) and format_choice != "COCO":
                 if isinstance(cln_results, dict) and "box_2d" in cln_results:
                     cln_results = [cln_results]
                 else:
                     st.error(f"Expected a list of detections in the JSON response, but got type: {type(cln_results)}")
                     st.text_area("Problematic JSON Response:", value=json.dumps(cln_results, indent=2), height=150)
                     return None


            h, w = image.shape[:2]

            if format_choice == "YOLO":
                yolo_results = []
                if not isinstance(cln_results, list):
                     st.error(f"YOLO conversion requires a list of detections, but got: {type(cln_results)}")
                     return None
                for item in cln_results:
                     if not isinstance(item, dict) or "box_2d" not in item or "label" not in item:
                         st.warning(f"Skipping item due to missing keys or wrong type: {item}")
                         continue
                     if not isinstance(item["box_2d"], list) or len(item["box_2d"]) != 4:
                         st.warning(f"Skipping item due to invalid 'box_2d' format: {item}")
                         continue

                     y1, x1, y2, x2 = item["box_2d"]
                     if not all(isinstance(c, (int, float)) and 0 <= c <= 1000 for c in [y1, x1, y2, x2]):
                         st.warning(f"Skipping item due to invalid coordinates (expected numbers 0-1000): {item['box_2d']}")
                         continue

                     abs_x1, abs_x2 = x1 / 1000 * w, x2 / 1000 * w
                     abs_y1, abs_y2 = y1 / 1000 * h, y2 / 1000 * h
                     x_center = (abs_x1 + abs_x2) / 2 / w
                     y_center = (abs_y1 + abs_y2) / 2 / h
                     width_norm = (abs_x2 - abs_x1) / w
                     height_norm = (abs_y2 - abs_y1) / h
                     x_center, y_center, width_norm, height_norm = np.clip([x_center, y_center, width_norm, height_norm], 0.0, 1.0)

                     if width_norm <= 0 or height_norm <= 0:
                         st.warning(f"Skipping item due to non-positive width/height after YOLO conversion: {item}")
                         continue

                     yolo_results.append({
                         "class": item["label"],
                         "bbox": [x_center, y_center, width_norm, height_norm]
                     })
                return yolo_results

            elif format_choice == "COCO":
                 coco_results = {
                     "annotations": [],
                     "images": [{"id": 1, "file_name": "image.jpg", "width": w, "height": h}],
                     "categories": []
                 }
                 category_map = {}
                 next_cat_id = 1
                 if not isinstance(cln_results, list):
                     st.error(f"COCO conversion expects a list of detections, but got: {type(cln_results)}")
                     return None

                 for idx, item in enumerate(cln_results):
                     if not isinstance(item, dict) or "box_2d" not in item or "label" not in item:
                         st.warning(f"Skipping item due to missing keys or wrong type: {item}")
                         continue
                     if not isinstance(item["box_2d"], list) or len(item["box_2d"]) != 4:
                         st.warning(f"Skipping item due to invalid 'box_2d' format: {item}")
                         continue

                     y1, x1, y2, x2 = item["box_2d"]
                     if not all(isinstance(c, (int, float)) and 0 <= c <= 1000 for c in [y1, x1, y2, x2]):
                         st.warning(f"Skipping item due to invalid coordinates (expected numbers 0-1000): {item['box_2d']}")
                         continue

                     abs_x1, abs_x2 = x1 / 1000 * w, x2 / 1000 * w
                     abs_y1, abs_y2 = y1 / 1000 * h, y2 / 1000 * h
                     abs_x1, abs_y1 = max(0, abs_x1), max(0, abs_y1)
                     abs_x2, abs_y2 = min(w, abs_x2), min(h, abs_y2)
                     coco_w = abs_x2 - abs_x1
                     coco_h = abs_y2 - abs_y1

                     if coco_w <= 0 or coco_h <= 0:
                         st.warning(f"Skipping item due to non-positive width/height after COCO conversion: {item}")
                         continue

                     label = item["label"]
                     if label not in category_map:
                         category_map[label] = next_cat_id
                         coco_results["categories"].append({"id": next_cat_id, "name": label, "supercategory": "object"})
                         next_cat_id += 1
                     category_id = category_map[label]

                     coco_results["annotations"].append({
                         "id": idx + 1,
                         "image_id": 1,
                         "category_id": category_id,
                         "bbox": [abs_x1, abs_y1, coco_w, coco_h],
                         "area": coco_w * coco_h,
                         "iscrowd": 0,
                         "segmentation": []
                     })
                 return coco_results

            else:
                if not isinstance(cln_results, list):
                     st.error(f"Expected a list of detections for JSON output, but got: {type(cln_results)}")
                     return None
                valid_results = []
                for item in cln_results:
                     if isinstance(item, dict) and "box_2d" in item and "label" in item:
                         if isinstance(item["box_2d"], list) and len(item["box_2d"]) == 4 and all(isinstance(c, (int, float)) for c in item["box_2d"]):
                              y1, x1, y2, x2 = item["box_2d"]
                              if all(0 <= c <= 1000 for c in [y1, x1, y2, x2]):
                                  valid_results.append(item)
                              else:
                                  st.warning(f"Skipping item due to invalid coordinates (expected 0-1000): {item}")
                         else:
                             st.warning(f"Skipping item due to invalid 'box_2d' format: {item}")
                     else:
                         st.warning(f"Skipping invalid item in JSON response (missing keys or wrong type): {item}")
                return valid_results


        except json.JSONDecodeError as json_e:
            st.error(f"Fatal Error decoding JSON from Gemini response: {str(json_e)}")
            st.text_area("Raw Gemini Response:", value=results, height=150)
            return None
        except Exception as e:
            st.error(f"Error processing results: {str(e)}")
            st.error(traceback.format_exc())
            return None

    def visualize_results(self, image: np.ndarray, results, format_choice: str) -> np.ndarray:
        """
        Draws bounding boxes and labels onto an image based on the processed
        detection results and the specified format.

        Args:
            image: The original image as a NumPy array.
            results: The processed annotations (list for JSON/YOLO, dict for COCO)
                     as returned by `process_detection_results`.
            format_choice: The format of the `results` ('JSON', 'YOLO', 'COCO').

        Returns:
            A NumPy array representing the image with annotations drawn, or the
            original image if `results` are empty or an error occurs.
        """
        try:
            annotator = Annotator(image.copy(), line_width=2, font_size=10)
            h, w = image.shape[:2]

            if not results:
                 st.warning("No valid results provided to visualize.")
                 return image

            if format_choice == "YOLO":
                if not isinstance(results, list):
                     st.error(f"Visualization error: Expected list for YOLO results, got {type(results)}")
                     return image
                for item in results:
                    if not isinstance(item, dict) or "bbox" not in item or "class" not in item:
                        st.warning(f"Skipping visualization for invalid YOLO item: {item}")
                        continue
                    if not isinstance(item["bbox"], list) or len(item["bbox"]) != 4:
                        st.warning(f"Skipping visualization due to invalid YOLO bbox format: {item}")
                        continue

                    x_center, y_center, width_norm, height_norm = item["bbox"]
                    box_w = width_norm * w
                    box_h = height_norm * h
                    x1 = (x_center * w) - box_w / 2
                    y1 = (y_center * h) - box_h / 2
                    x2 = x1 + box_w
                    y2 = y1 + box_h
                    x1, y1, x2, y2 = map(int, [max(0, x1), max(0, y1), min(w - 1, x2), min(h - 1, y2)])
                    if x1 < x2 and y1 < y2:
                        annotator.box_label([x1, y1, x2, y2], label=item["class"], color=colors(0, True))
                    else:
                        st.warning(f"Skipping visualization for degenerate YOLO box: {item}")


            elif format_choice == "COCO":
                 if not isinstance(results, dict) or "annotations" not in results or "categories" not in results:
                     st.error(f"Visualization error: Invalid COCO format provided: {type(results)}")
                     return image

                 cat_id_to_name = {cat['id']: cat['name'] for cat in results.get('categories', [])}

                 for ann in results.get("annotations", []):
                     if not isinstance(ann, dict) or "bbox" not in ann or "category_id" not in ann:
                          st.warning(f"Skipping visualization for invalid COCO annotation: {ann}")
                          continue
                     if not isinstance(ann["bbox"], list) or len(ann["bbox"]) != 4:
                         st.warning(f"Skipping visualization due to invalid COCO bbox format: {ann}")
                         continue

                     x1, y1, box_w, box_h = ann["bbox"]
                     label = cat_id_to_name.get(ann["category_id"], "unknown")
                     x1, y1 = max(0, x1), max(0, y1)
                     x2 = min(w - 1, x1 + box_w)
                     y2 = min(h - 1, y1 + box_h)
                     x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])

                     if x1 < x2 and y1 < y2:
                         annotator.box_label([x1, y1, x2, y2], label=label, color=colors(ann.get("category_id", 0), True))
                     else:
                         st.warning(f"Skipping visualization for degenerate COCO box: {ann}")


            else:
                if not isinstance(results, list):
                     st.error(f"Visualization error: Expected list for JSON results, got {type(results)}")
                     return image
                for idx, item in enumerate(results):
                     if not isinstance(item, dict) or "box_2d" not in item or "label" not in item:
                         st.warning(f"Skipping visualization for invalid JSON item: {item}")
                         continue
                     if not isinstance(item["box_2d"], list) or len(item["box_2d"]) != 4:
                         st.warning(f"Skipping visualization due to invalid JSON box_2d format: {item}")
                         continue

                     y1_norm, x1_norm, y2_norm, x2_norm = item["box_2d"]
                     abs_y1 = y1_norm / 1000 * h
                     abs_x1 = x1_norm / 1000 * w
                     abs_y2 = y2_norm / 1000 * h
                     abs_x2 = x2_norm / 1000 * w
                     abs_x1, abs_y1 = max(0, abs_x1), max(0, abs_y1)
                     abs_x2, abs_y2 = min(w - 1, abs_x2), min(h - 1, abs_y2)
                     abs_x1, abs_y1, abs_x2, abs_y2 = map(int, [abs_x1, abs_y1, abs_x2, abs_y2])

                     if abs_x1 < abs_x2 and abs_y1 < abs_y2:
                         annotator.box_label([abs_x1, abs_y1, abs_x2, abs_y2], label=item["label"], color=colors(idx, True))
                     else:
                         st.warning(f"Skipping visualization for degenerate JSON box: {item}")

            return annotator.result()

        except Exception as e:
            st.error(f"Error visualizing results: {str(e)}")
            st.error(traceback.format_exc())
            return image

    def process_single_image(self, image_bytes: bytes, filename: str, custom_prompt: str, format_choice: str):
        """Processes a single image (bytes), calls Gemini, and returns results."""
        try:
            image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
            image_np = np.array(image)

            task_prompt = f"Your task is to perform object detection. {custom_prompt}."
            output_prompt = (
                "Respond ONLY with a valid JSON list containing objects. Each object must have "
                "'label' (string) and 'box_2d' (list of 4 numbers representing [ymin, xmin, ymax, xmax] "
                "normalized from 0 to 1000). Do not include any other text, explanations, or markdown formatting like ```json."
                " Example: [{'label': 'car', 'box_2d': [100, 200, 300, 400]}, {'label': 'person', 'box_2d': [500, 600, 700, 800]}]"
             )
            full_prompt = task_prompt + " " + output_prompt

            result_data = {
                "filename": filename,
                "image_bytes": image_bytes,
                "image_np": image_np,
                "raw_response": None,
                "annotations": None,
                "format": format_choice,
                "error": None
            }

            try:
                response = self.model.generate_content(
                    [full_prompt, image],
                    generation_config=genai.GenerationConfig(temperature=self.temperature),
                    request_options={'timeout': 120}
                )

                if not response.parts:
                    if response.prompt_feedback.block_reason:
                        error_msg = f"Request blocked for {filename} due to: {response.prompt_feedback.block_reason.name}"
                        st.error(error_msg)
                        result_data["raw_response"] = "BLOCKED"
                        result_data["error"] = error_msg
                    else:
                        error_msg = f"Received an empty response for {filename}."
                        st.error(error_msg)
                        result_data["raw_response"] = "EMPTY"
                        result_data["error"] = error_msg
                else:
                    result_data["raw_response"] = response.text
                    processed_annotations = self.process_detection_results(response.text, image_np, format_choice)
                    result_data["annotations"] = processed_annotations
                    if processed_annotations is None:
                        result_data["error"] = f"Failed to process annotations for {filename}"


            except types.generation_types.BlockedPromptException as bpe:
                error_msg = f"Request blocked by API for {filename}: {bpe}"
                st.error(error_msg)
                result_data["raw_response"] = "BLOCKED"
                result_data["error"] = error_msg
            except types.generation_types.StopCandidateException as sce:
                 error_msg = f"Generation stopped unexpectedly for {filename}: {sce}"
                 st.error(error_msg)
                 result_data["raw_response"] = "STOPPED"
                 result_data["error"] = error_msg
            except Exception as e:
                 error_msg = f"Error during inference for {filename}: {str(e)}"
                 st.error(error_msg)
                 result_data["raw_response"] = f"ERROR: {e}"
                 result_data["error"] = error_msg

            return result_data

        except Exception as img_proc_e:
            error_msg = f"Error preparing image {filename}: {str(img_proc_e)}"
            st.error(error_msg)
            return {
                "filename": filename,
                "image_bytes": image_bytes,
                "image_np": None,
                "raw_response": f"IMAGE_PREP_ERROR: {img_proc_e}",
                "annotations": None,
                "format": format_choice,
                "error": error_msg
            }

