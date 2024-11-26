import os
import subprocess
import json
from datetime import datetime

AWS_ACCESS = os.getenv('AWS_BASEBALLCV_ACCESS_KEY')
AWS_SECRET = os.getenv('AWS_BASEBALLCV_SECRET_KEY')
AWS_REG = os.getenv('AWS_BASEBALLCV_REGION')

class S3Manager:
    """
    A class to manage interactions with Amazon S3, including creating buckets,
    creating folders, uploading files, and managing annotation and task queues using AWS CLI.
    
    Attributes:
        bucket_name (str): The name of the S3 bucket to manage.
    """
    
    def __init__(self, bucket_name: str) -> None:
        """
        Initialize the S3Manager with a specified bucket name.
        
        Args:
            bucket_name (str): The name of the S3 bucket to use.
        """
        self.bucket_name = bucket_name

    def _run_cli_command(self, command: list) -> None:
        """
        Run an AWS CLI command using subprocess.
        
        Args:
            command (list): List of command-line arguments for the AWS CLI command.
        
        Raises:
            Exception: If the AWS CLI command fails.
        """
        try:
            result = subprocess.run(command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            print(result.stdout.decode())
        except subprocess.CalledProcessError as e:
            print(f"Error running AWS CLI command: {e.stderr.decode()}")
            raise Exception(f"Error: {e.stderr.decode()}")

    def create_bucket(self) -> None:
        """
        Create an S3 bucket if it does not already exist using AWS CLI.
        """
        try:
            if AWS_REG == "us-east-1":
                command = ["aws", "s3api", "create-bucket", "--bucket", self.bucket_name]
            else:
                command = [
                    "aws", "s3api", "create-bucket", "--bucket", self.bucket_name,
                    "--create-bucket-configuration", f"LocationConstraint={AWS_REG}"
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
        s3_key = f"{folder_name}/"
        command = ["aws", "s3", "api", "put-object", "--bucket", self.bucket_name, "--key", s3_key]
        self._run_cli_command(command)
        print(f"Folder '{folder_name}' created in bucket '{self.bucket_name}'.")

    def upload_file(self, folder_name: str, file_path: str) -> None:
        """
        Upload a file to the specified S3 bucket and folder using AWS CLI.
        
        Args:
            folder_name (str): The name of the folder to upload the file to.
            file_path (str): The local path of the file to upload.
        """
        file_name = os.path.basename(file_path)
        s3_key = f"{folder_name}/{file_name}"
        command = ["aws", "s3", "cp", file_path, f"s3://{self.bucket_name}/{s3_key}"]
        self._run_cli_command(command)
        print(f"File '{file_name}' uploaded to '{self.bucket_name}/{s3_key}'.")

    def upload_annotation_data(self, folder_name: str, annotations: dict) -> None:
        """
        Upload annotation data in JSON format to the specified S3 folder using AWS CLI.
        
        Args:
            folder_name (str): The folder in the S3 bucket.
            annotations (dict): The annotation data to be uploaded.
        """
        json_data = json.dumps(annotations, indent=4)
        temp_file = "/tmp/annotations.json"
        with open(temp_file, "w") as f:
            f.write(json_data)
        
        s3_key = f"{folder_name}/annotations.json"
        command = ["aws", "s3", "cp", temp_file, f"s3://{self.bucket_name}/{s3_key}"]
        self._run_cli_command(command)
        os.remove(temp_file)
        print(f"Annotations uploaded to '{self.bucket_name}/{s3_key}'.")

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
        # Ensure we're using just the filename part of the s3_key
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
        Retrieve raw photos from a folder in the S3 bucket and download them locally using AWS CLI.
        
        Args:
            s3_folder_name (str): The folder in S3 to retrieve photos from.
            local_path (str): The local directory path where photos will be saved.
            max_images (int): Maximum number of images to be downloaded into directory.
            
        Returns:
            list[str]: List of local paths to downloaded photos.
        """
        os.makedirs(local_path, exist_ok=True)
        
        if s3_folder_name and not s3_folder_name.endswith('/'):
            s3_folder_name += '/'
        
        #command = ["aws", "s3", "ls", f"s3://{self.bucket_name}/{s3_folder_name}"]
        #result = subprocess.run(command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        command = ["aws", "s3", "ls", f"s3://{self.bucket_name}/{s3_folder_name}"]
        result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if result.returncode != 0:
            print("Error:", result.stderr.decode())
        files = result.stdout.decode().splitlines()



        files = result.stdout.decode().splitlines()
        
        downloaded_files = []
        for file in files:
            parts = file.split()
            if len(parts) >= 4:
                filename = parts[3]
                if filename.lower().endswith(('.jpg', '.jpeg')):
                    s3_key = f"{s3_folder_name}{filename}"
                    local_file_path = os.path.join(local_path, filename)
                    
                    if self._download_file(s3_key, local_file_path):
                        downloaded_files.append(local_file_path)
                        
                    if len(downloaded_files) >= max_images:
                        break
        
        return downloaded_files
