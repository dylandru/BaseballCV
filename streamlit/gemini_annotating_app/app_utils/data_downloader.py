import streamlit as st
import base64
import io
import json
import zipfile
import yaml
from pathlib import Path
import traceback
from typing import List, Dict, Any, Optional, Set, Tuple, IO

class DataDownloader:
    """
    Handles the formatting and packaging of image annotation results into
    downloadable ZIP archives based on user-selected formats (COCO, YOLO, JSON).
    """

    def _format_yolo_txt(self, processed_results: List[Dict[str, Any]], class_to_id: Dict[str, int]) -> str:
        """
        Formats a list of detection results for a single image into the YOLO .txt format.

        Each line in the output string represents one detected object in the format:
        `<class_id> <x_center_norm> <y_center_norm> <width_norm> <height_norm>`
        Coordinates normalized.

        Args:
            processed_results (list): A list of dictionaries, where each dictionary
                               represents a detected object and should contain 'class' (label string)
                               and 'bbox' (list/tuple of normalized [x_center, y_center, width, height]).
            class_to_id (dict): A dictionary mapping class label strings to integer class IDs.

        Returns:
            str: A string formatted according to YOLO annotation standards, with one line per valid detection.
        """
        lines = []
        if isinstance(processed_results, list):
            for item in processed_results:
                label = item.get("class")
                bbox = item.get("bbox")
                if label and bbox and label in class_to_id and len(bbox) == 4:
                    class_id = class_to_id[label]
                    try:
                        lines.append(f"{class_id} {float(bbox[0]):.6f} {float(bbox[1]):.6f} {float(bbox[2]):.6f} {float(bbox[3]):.6f}")
                    except (ValueError, TypeError):
                         st.warning(f"Skipping invalid bbox data for label '{label}': {bbox}")
        return "\n".join(lines)

    def _create_data_yaml(self, class_names: List[str]) -> str:
        """
        Generates the content for a `data.yaml` file required by YOLO training pipelines. Includes the number of classes (`nc`) and a list of class names (`names`).

        Args:
            class_names (list): A sorted list of unique class label strings.

        Returns:
            str: A string containing the YAML representation of the data configuration.
        """
        data = {
            'names': class_names,
            'nc': len(class_names),
            'train': '../train/images',
            'val': '../val/images'
        }
        return yaml.dump(data, default_flow_style=False)

    def _merge_coco_results(self, all_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Merges annotation results from multiple images into a single COCO JSON structure.

        Args:
            all_results (list): A list of dictionaries. Each dictionary represents the
                         processed result for one image and is expected to contain
                         'filename', 'image_bytes', 'format' ('COCO'), and 'annotations'
                         (a dict following COCO structure for a single image).

        Returns:
            dict: Dictionary representing the merged COCO dataset, containing
            'images', 'annotations', and 'categories' keys populated with
            data from all valid input results.
        """
        merged_coco: Dict[str, List[Any]] = {
            "images": [],
            "annotations": [],
            "categories": []
        }
        annotation_id_counter: int = 1
        category_map: Dict[str, int] = {}
        next_cat_id: int = 1
        processed_image_id: int = 1

        for result in all_results:
            if not isinstance(result, dict):
                st.warning(f"Skipping invalid result entry (expected dict): {type(result)}")
                continue

            filename = result.get('filename')
            if not filename:
                 st.warning("Skipping result due to missing filename.")
                 continue

            if result.get('format') != 'COCO':
                 st.warning(f"Skipping result for '{filename}' as it's not in COCO format during merge.")
                 continue

            coco_data = result.get('annotations')
            if not coco_data or not isinstance(coco_data, dict):
                st.warning(f"Skipping result for '{filename}' due to missing or invalid COCO 'annotations' data.")
                continue

            image_info_list = coco_data.get('images', [])
            if not image_info_list or not isinstance(image_info_list, list) or len(image_info_list) != 1:
                st.warning(f"Missing, invalid, or unexpected 'images' field in COCO data for '{filename}'. Expected a list with one entry.")
                continue
            image_info = image_info_list[0]
            if not isinstance(image_info, dict):
                 st.warning(f"Invalid image info entry (expected dict) in COCO data for '{filename}'.")
                 continue

            image_info['id'] = processed_image_id
            image_info['file_name'] = filename
            merged_coco["images"].append(image_info)

            original_categories = coco_data.get('categories', [])
            if not isinstance(original_categories, list):
                 st.warning(f"Invalid 'categories' field (expected list) in COCO data for '{filename}'. Annotations for this image might be skipped.")
            else:
                for cat in original_categories:
                    if isinstance(cat, dict) and 'name' in cat:
                        cat_name = cat['name']
                        if cat_name not in category_map:
                            category_map[cat_name] = next_cat_id
                            merged_coco["categories"].append({
                                "id": next_cat_id,
                                "name": cat_name,
                                "supercategory": cat.get('supercategory', 'object')
                            })
                            next_cat_id += 1
                    else:
                         st.warning(f"Skipping invalid category entry: {cat} in '{filename}'")

            original_annotations = coco_data.get('annotations', [])
            if not isinstance(original_annotations, list):
                 st.warning(f"Invalid 'annotations' field (expected list) in COCO data for '{filename}'. Skipping annotations for this image.")
                 processed_image_id += 1
                 continue

            for ann in original_annotations:
                 if not isinstance(ann, dict):
                     st.warning(f"Skipping invalid annotation entry (expected dict): {ann} in '{filename}'")
                     continue

                 original_cat_id = ann.get('category_id')
                 if original_cat_id is None:
                      st.warning(f"Skipping annotation (ID: {ann.get('id', 'N/A')}) due to missing 'category_id' in '{filename}'")
                      continue

                 original_cat_name = None
                 for c in original_categories:
                      if isinstance(c, dict) and c.get('id') == original_cat_id:
                          original_cat_name = c.get('name')
                          break

                 if original_cat_name and original_cat_name in category_map:
                     ann['id'] = annotation_id_counter
                     ann['image_id'] = processed_image_id
                     ann['category_id'] = category_map[original_cat_name]
                     merged_coco["annotations"].append(ann)
                     annotation_id_counter += 1
                 else:
                      st.warning(f"Could not map category ID {original_cat_id} (name: '{original_cat_name or 'Not Found'}') for annotation {ann.get('id', 'N/A')} from image '{filename}'. Annotation skipped.")

            processed_image_id += 1

        merged_coco["categories"] = sorted(merged_coco["categories"], key=lambda x: x['id'])

        return merged_coco

    def create_zip_file(self, results_list: List[Dict[str, Any]]) -> Optional[IO[bytes]]:
        """
        Creates a ZIP archive in memory containing images and their corresponding annotations.

        The structure and format of the annotations within the ZIP file depend on the
        'format' specified in the input results (detected from the first valid result).
        Supported formats:
        - 'COCO': Creates a single `_annotations.coco.json` file with merged annotations
                  and an `images/` directory containing all image files.
        - 'YOLO': Creates a `labels/` directory with a `.txt` file per image, an
                  `images/` directory with `.jpg` images, and a `data.yaml` file
                  at the root.
        - 'JSON' (Default): Creates an `annotations/` directory with a `.json` file
                  per image and an `images/` directory with `.jpg` images.

        Args:
            results_list (list): A list of dictionaries, where each dictionary represents
                          the processed result for a single image. Expected keys include
                          'filename' (str), 'image_bytes' (bytes), 'annotations' (format-dependent),
                          and 'format' (str, e.g., 'COCO', 'YOLO', 'JSON').

        Returns:
            Optional[IO[bytes]]: An in-memory bytes buffer (BytesIO) containing the ZIP file content,
            ready for download.
        """
        if not results_list:
            st.error("Cannot create ZIP file: No results provided.")
            return None

        zip_buffer = io.BytesIO()
        output_format: Optional[str] = None
        for r in results_list:
             if r and isinstance(r, dict) and r.get('format'):
                 output_format = r['format']
                 break

        if not output_format:
             st.error("Cannot determine output format from results list.")
             return None

        st.info(f"Creating ZIP file with format: {output_format}")

        class_names: List[str] = []
        class_to_id: Dict[str, int] = {}
        if output_format == 'YOLO':
            unique_labels: Set[str] = set()
            for result in results_list:
                annotations = result.get('annotations')
                if isinstance(annotations, list):
                    for item in annotations:
                        if isinstance(item, dict) and 'class' in item and item['class'] is not None:
                            unique_labels.add(item['class'])
                elif annotations is not None:
                     filename = result.get('filename', 'Unknown file')
                     st.warning(f"Annotations for '{filename}' is not a list: {type(annotations)}. Skipping for YOLO label collection.")

            class_names = sorted(list(unique_labels))
            class_to_id = {name: i for i, name in enumerate(class_names)}
            if not class_names:
                 st.warning("No valid labels found across all results for YOLO format. `data.yaml` will be omitted, and .txt files might be empty.")

        try:
            with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED, False) as zip_file:
                if output_format == 'COCO':
                    coco_results_list = [r for r in results_list if isinstance(r, dict) and r.get('format') == 'COCO']
                    if not coco_results_list:
                         st.warning("No valid COCO results found to include in the ZIP.")
                    else:
                        merged_coco_data = self._merge_coco_results(coco_results_list)
                        if merged_coco_data and merged_coco_data.get('images'):
                            zip_file.writestr('_annotations.coco.json', json.dumps(merged_coco_data, indent=2))
                            filenames_in_coco = {img['file_name'] for img in merged_coco_data['images']}
                            added_images = set()
                            for result in coco_results_list:
                                filename = result.get('filename')
                                img_bytes = result.get('image_bytes')
                                if filename and img_bytes and filename in filenames_in_coco and filename not in added_images:
                                     zip_file.writestr(f"images/{filename}", img_bytes)
                                     added_images.add(filename)
                                elif filename and filename in filenames_in_coco and not img_bytes:
                                    st.warning(f"Missing image bytes for '{filename}', cannot add image to COCO zip.")
                                elif filename and filename not in filenames_in_coco:
                                     st.warning(f"Image '{filename}' was present in input but not included in final COCO JSON, skipping image file.")
                        else:
                            st.warning("Merging COCO data failed or resulted in no images. No COCO annotation file or images added.")

                elif output_format == 'YOLO':
                    if class_names:
                        yaml_content = self._create_data_yaml(class_names)
                        zip_file.writestr('data.yaml', yaml_content)
                    else:
                         st.warning("Skipping data.yaml creation as no class names were found.")

                    labels_folder = 'labels'
                    images_folder = 'images'
                    for result in results_list:
                        if not isinstance(result, dict): continue
                        annotations = result.get('annotations')
                        img_bytes = result.get('image_bytes')
                        filename = result.get('filename')

                        if img_bytes and filename:
                            filename_stem = Path(filename).stem
                            img_filename = f"{filename_stem}.jpg"
                            txt_filename = f"{filename_stem}.txt"

                            zip_file.writestr(f"{images_folder}/{img_filename}", img_bytes)

                            if isinstance(annotations, list):
                                yolo_txt = self._format_yolo_txt(annotations, class_to_id)
                                zip_file.writestr(f"{labels_folder}/{txt_filename}", yolo_txt)
                            else:
                                zip_file.writestr(f"{labels_folder}/{txt_filename}", "")
                                if annotations is not None:
                                     st.warning(f"Annotations for '{filename}' are not a list ({type(annotations)}). Creating empty .txt file.")
                        elif not filename:
                             st.warning("Skipping YOLO output for an entry: missing filename.")
                        elif not img_bytes:
                             st.warning(f"Skipping YOLO output for '{filename}': missing image bytes.")

                else: # JSON format
                    images_folder = 'images'
                    annotations_folder = 'annotations'
                    for result in results_list:
                        if not isinstance(result, dict): continue
                        annotations = result.get('annotations')
                        img_bytes = result.get('image_bytes')
                        filename = result.get('filename')

                        if img_bytes and filename:
                            filename_stem = Path(filename).stem
                            img_filename = f"{filename_stem}.jpg"
                            json_filename = f"{filename_stem}.json"

                            zip_file.writestr(f"{images_folder}/{img_filename}", img_bytes)

                            if annotations is not None:
                                try:
                                    json_content = json.dumps(annotations, indent=2)
                                    zip_file.writestr(f"{annotations_folder}/{json_filename}", json_content)
                                except TypeError as json_e:
                                     st.warning(f"Could not serialize annotations to JSON for '{filename}': {json_e}. Skipping annotation file.")
                            else:
                                 st.info(f"Annotations for '{filename}' are None. Skipping JSON annotation file.")

                        elif not filename:
                             st.warning("Skipping JSON output for an entry: missing filename.")
                        elif not img_bytes:
                             st.warning(f"Skipping JSON output for '{filename}': missing image bytes.")

        except Exception as zip_e:
            st.error(f"Failed to create ZIP file content: {zip_e}")
            st.error(traceback.format_exc())
            return None

        zip_buffer.seek(0)
        return zip_buffer

    def get_zip_download_link(self, zip_buffer: Optional[IO[bytes]], filename: str = "annotations_batch.zip") -> str:
        """
        Generates a Streamlit download link for the provided ZIP buffer.

        Args:
            zip_buffer (Optional[IO[bytes]]): The in-memory bytes buffer containing the ZIP file content.
            filename (str): The desired filename for the downloaded ZIP file.

        Returns:
            str: An HTML string representing the download link, or an empty string if
            the buffer is invalid or an error occurs during link generation.
        """
        if zip_buffer is None:
             st.error("Download link cannot be created: ZIP buffer is invalid.")
             return ""
        try:
            zip_buffer.seek(0)
            zip_bytes = zip_buffer.getvalue()
            b64 = base64.b64encode(zip_bytes).decode()
            href = f'<a href="data:application/zip;base64,{b64}" download="{filename}">Download All as ZIP ({filename})</a>'
            return href
        except Exception as e:
             st.error(f"Failed to create download link: {e}")
             st.error(traceback.format_exc())
             return ""
