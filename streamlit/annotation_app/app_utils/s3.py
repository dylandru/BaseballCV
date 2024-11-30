import os
import subprocess
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
import tempfile

class S3Manager:
    """
    A class to manage interactions with Amazon S3, including creating buckets,
    creating folders, uploading files, and managing annotation and task queues using AWS CLI.
    
    Attributes:
        bucket_name (str): The name of the S3 bucket to manage.
    """
    
    def __init__(self, bucket_name: str, max_workers: int = 10) -> None:
        """
        Initialize the S3Manager with a specified bucket name.
        
        Args:
            bucket_name (str): The name of the S3 bucket to use.
        """
        self.bucket_name = bucket_name
        self.env = {
            "AWS_ACCESS_KEY_ID": os.getenv('AWS_BASEBALLCV_ACCESS_KEY'),
            "AWS_SECRET_ACCESS_KEY": os.getenv('AWS_BASEBALLCV_SECRET_KEY'),
            "AWS_DEFAULT_REGION": os.getenv('AWS_BASEBALLCV_REGION'),
            **os.environ
        }
        self.max_workers = max_workers
        if not all([self.env["AWS_ACCESS_KEY_ID"], 
                   self.env["AWS_SECRET_ACCESS_KEY"], 
                   self.env["AWS_DEFAULT_REGION"]]):
            raise ValueError("AWS credentials not properly configured. Please set AWS_BASEBALLCV_ACCESS_KEY, AWS_BASEBALLCV_SECRET_KEY, and AWS_BASEBALLCV_REGION environment variables.")

    def _run_cli_command(self, command: list) -> None:
        """
        Run an AWS CLI command using subprocess with credentials from environment variables.
        
        Args:
            command (list): List of command-line arguments for the AWS CLI command.
        """
        try:
            result = subprocess.run(
                command,
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                env=self.env
            )
            print(f"Command output: {result.stdout.decode()}")
        except subprocess.CalledProcessError as e:
            error_msg = e.stderr.decode()
            print(f"Error running AWS CLI command: {error_msg}")
            raise RuntimeError(f"Error: {error_msg}")

    def create_bucket(self) -> None:
        """Create an S3 bucket if it does not already exist using AWS CLI."""
        try:
            if self.env["AWS_DEFAULT_REGION"] == "us-east-1":
                command = ["aws", "s3api", "create-bucket", "--bucket", self.bucket_name]
            else:
                command = [
                    "aws", "s3api", "create-bucket", "--bucket", self.bucket_name,
                    "--create-bucket-configuration", f"LocationConstraint={self.env['AWS_DEFAULT_REGION']}"
                ]
            self._run_cli_command(command)
            print(f"Bucket '{self.bucket_name}' created successfully.")
        except Exception as e:
            print(f"Error creating bucket: {e}")

    def create_folder(self, folder_name: str) -> None:
        """
        Create a folder in the specified S3 bucket using AWS CLI.
        
        Args:
            folder_name (str): The name of the folder to create.
        """
        if not folder_name:
            raise ValueError("Folder name must not be empty or None.")

        folder_path = folder_name.strip('/')
        if not folder_path:
            raise ValueError("Invalid folder path after normalization")

        dummy_file_for_folder = os.path.join(os.getcwd(), 'empty.txt')
        try:
            with open(dummy_file_for_folder, 'w') as f:
                pass

            s3_path = f"s3://{self.bucket_name}/{folder_path}/.keep"
            command = ["aws", "s3", "cp", dummy_file_for_folder, s3_path]
            
            self._run_cli_command(command)
            print(f"Folder '{folder_path}' created successfully in bucket '{self.bucket_name}'")
            
        except Exception as e:
            raise RuntimeError(f"Failed to create folder '{folder_path}': {e}")
        finally:
            if os.path.exists(dummy_file_for_folder):
                os.remove(dummy_file_for_folder)

    def upload_file(self, folder_name: str, file_path: str) -> None:
        """
        Upload a file to the specified S3 bucket and folder using AWS CLI.
        The file remains in its original location after upload.
        
        Args:
            folder_name (str): The name of the folder to upload the file to.
            file_path (str): The local path of the file to upload.
        """
        if not os.path.exists(file_path):
            raise ValueError(f"File not found: {file_path}")

        folder_path = folder_name.strip('/')
        file_name = os.path.basename(file_path)
        s3_key = f"{folder_path}/{file_name}" if folder_path else file_name
        
        command = ["aws", "s3", "cp", file_path, f"s3://{self.bucket_name}/{s3_key}"]
        try:
            self._run_cli_command(command)
            print(f"File '{file_name}' uploaded to '{self.bucket_name}/{s3_key}'")
        except Exception as e:
            raise RuntimeError(f"Failed to upload file '{file_name}': {e}")

    def copy_file(self, source_key: str, dest_key: str) -> None:
        """
        Copy a file within the S3 bucket.
        
        Args:
            source_key (str): Source path in the bucket
            dest_key (str): Destination path in the bucket
        """
        command = ["aws", "s3", "cp", 
                  f"s3://{self.bucket_name}/{source_key}",
                  f"s3://{self.bucket_name}/{dest_key}"]
        try:
            self._run_cli_command(command)
            print(f"Copied '{source_key}' to '{dest_key}'")
        except Exception as e:
            raise RuntimeError(f"Failed to copy file: {e}")

    def download_file(self, s3_key: str, local_path: str) -> None:
        """
        Download a file from S3 to a local path.
        
        Args:
            s3_key (str): The key (path) of the file in S3
            local_path (str): The local path where the file should be saved
        """
        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        command = ["aws", "s3", "cp", f"s3://{self.bucket_name}/{s3_key}", local_path]
        try:
            self._run_cli_command(command)
            print(f"Downloaded '{s3_key}' to '{local_path}'")
        except Exception as e:
            raise RuntimeError(f"Failed to download file: {e}")

    def upload_annotation_data(self, folder_name: str, annotations: dict) -> None:
        """
        Upload annotation data in JSON format to the specified S3 folder using AWS CLI.
        
        Args:
            folder_name (str): The folder in the S3 bucket.
            annotations (dict): The annotation data to be uploaded.
        """
        json_data = json.dumps(annotations, indent=4)
        temp_file = os.path.join(os.getcwd(), 'temp_annotations.json')
        try:
            with open(temp_file, "w") as f:
                f.write(json_data)
            
            s3_key = f"{folder_name}/annotations.json"
            command = ["aws", "s3", "cp", temp_file, f"s3://{self.bucket_name}/{s3_key}"]
            self._run_cli_command(command)
            print(f"Annotations uploaded to '{self.bucket_name}/{s3_key}'")
        except Exception as e:
            raise RuntimeError(f"Failed to upload annotations: {e}")
        finally:
            if os.path.exists(temp_file):
                os.remove(temp_file)

    def update_task_queue(self, folder_name: str, task_queue: dict) -> None:
        """
        Update task queue data in JSON format to the specified S3 folder using AWS CLI.
        
        Args:
            folder_name (str): The folder in the S3 bucket.
            task_queue (dict): The task queue data to be uploaded.
        """
        json_data = json.dumps(task_queue, indent=4)
        temp_file = "/tmp/task_queue.json"
        with open(temp_file, "w") as f:
            f.write(json_data)
        
        s3_key = f"{folder_name}/task_queue.json"
        command = ["aws", "s3", "cp", temp_file, f"s3://{self.bucket_name}/{s3_key}"]
        self._run_cli_command(command)
        os.remove(temp_file)
        print(f"Task queue updated in '{self.bucket_name}/{s3_key}'.")

    def _download_file(self, s3_key: str, local_path: str) -> bool:
        """
        Download a file from the S3 bucket to the local file system using AWS CLI.
        
        Args:
            s3_key (str): The S3 key of the file to download.
            local_path (str): The local path where the file will be saved.
            
        Returns:
            bool: True if download was successful, False otherwise.
        """
        clean_key = s3_key.split()[-1] if len(s3_key.split()) > 1 else s3_key
        command = ["aws", "s3", "cp", f"s3://{self.bucket_name}/{clean_key}", local_path]
        try:
            self._run_cli_command(command)
            print(f"File '{clean_key}' downloaded to '{local_path}'.")
            return True
        except Exception as e:
            print(f"Error downloading {clean_key}: {e}")
            return False
       
    def retrieve_raw_photos(self, s3_folder_name: str, local_path: str, max_images: int) -> list[str]:
        """
        Retrieve raw photos from a folder in the S3 bucket and download them locally using AWS CLI with
        parallel downloading for better performance.
        
        Args:
            s3_folder_name (str): The folder in S3 to retrieve photos from.
            local_path (str): The local directory path where photos will be saved.
            max_images (int): Maximum number of images to be downloaded into directory.
            
        Returns:
            list[str]: List of local paths to downloaded photos.
        """
        os.makedirs(local_path, exist_ok=True)
        
        command = [
            "aws", "s3", "ls", f"s3://{self.bucket_name}/{s3_folder_name}",
            "--recursive",
            "|", "grep", "-i", "'.jpg\|.jpeg'",
            "|", "head", f"-n {max_images}"
        ]
        
        result = subprocess.run(
            " ".join(command), 
            shell=True, 
            stdout=subprocess.PIPE, 
            stderr=subprocess.PIPE,
            text=True
        )
        
        if result.returncode != 0:
            print(f"Error listing files: {result.stderr}")
            return []
            
        download_tasks = []
        files = result.stdout.splitlines()
        
        for file in files:
            parts = file.split()
            if len(parts) >= 4:
                filename = parts[3]
                if filename.lower().endswith(('.jpg', '.jpeg')):
                    local_file_path = os.path.join(local_path, os.path.basename(filename))
                    download_tasks.append((filename, local_file_path))
                    
                    if len(download_tasks) >= max_images:
                        break
        
        downloaded_files = []
        with ThreadPoolExecutor(max_workers=min(self.max_workers, len(download_tasks))) as executor:
            future_to_path = {
                executor.submit(self._download_file, s3_key, local_path): local_path
                for s3_key, local_path in download_tasks
            }
            
            for future in as_completed(future_to_path):
                local_path = future_to_path[future]
                try:
                    if future.result():
                        downloaded_files.append(local_path)
                except Exception as e:
                    print(f"Error downloading to {local_path}: {str(e)}")
        
        return downloaded_files

    def create_project_structure(self, project_name: str) -> None:
        """
        Create the initial project structure in S3.
        
        Args:
            project_name (str): Name of the project to create structure for.
        """
        base_folders = [
            f"{project_name}/images",
            f"{project_name}/completed",
            f"{project_name}/annotations"
        ]
        
        for folder in base_folders:
            self.create_folder(folder)
            
    def move_file(self, source_key: str, dest_key: str) -> None:
        """
        Move/Copy a file within the S3 bucket.
        
        Args:
            source_key (str): Source path in the bucket
            dest_key (str): Destination path in the bucket
        """
        command = ["aws", "s3", "mv", 
                  f"s3://{self.bucket_name}/{source_key}",
                  f"s3://{self.bucket_name}/{dest_key}"]
        try:
            self._run_cli_command(command)
            print(f"Moved '{source_key}' to '{dest_key}'")
        except Exception as e:
            raise RuntimeError(f"Failed to move file: {e}")

    def delete_file(self, s3_key: str) -> None:
        """
        Delete a file from the S3 bucket.
        
        Args:
            s3_key (str): The key (path) of the file to delete in S3
        """
        command = ["aws", "s3", "rm", f"s3://{self.bucket_name}/{s3_key}"]
        try:
            self._run_cli_command(command)
            print(f"Deleted '{s3_key}' from bucket '{self.bucket_name}'")
        except Exception as e:
            raise RuntimeError(f"Failed to delete file '{s3_key}': {e}")

    def upload_json_data(self, s3_key: str, data: dict) -> None:
        """
        Upload JSON data directly to S3 using a temporary file.
        
        Args:
            s3_key (str): The S3 key where the JSON will be uploaded
            data (dict): The data to be uploaded as JSON
        """
        with tempfile.NamedTemporaryFile(mode='w', delete=True) as tmp:
            json.dump(data, tmp)
            tmp.flush()
            command = ["aws", "s3", "cp", tmp.name, f"s3://{self.bucket_name}/{s3_key}"]
            self._run_cli_command(command)
