import os
import boto3
from botocore.exceptions import ClientError
import json

AWS_ACCESS = os.getenv('AWS_ACCESS_KEY')
AWS_SECRET = os.getenv('AWS_SECRET_KEY')
AWS_REG = os.getenv('AWS_REGION')

class S3Manager:
    """
    A class to manage interactions with Amazon S3, including creating buckets,
    creating folders, uploading files, and managing annotation and task queues.
    
    Attributes:
        s3_client (boto3.client): The S3 client used to interact with AWS S3.
        bucket_name (str): The name of the S3 bucket to manage.
    """
    
    def __init__(self, bucket_name: str) -> None:
        """
        Initialize the S3Manager with a specified bucket name.
        
        Args:
            bucket_name (str): The name of the S3 bucket to use.
        """
        self.bucket_name = bucket_name
        self.s3_client = boto3.client(
            "s3",
            aws_access_key_id=AWS_ACCESS,
            aws_secret_access_key=AWS_SECRET,
            region_name=AWS_REG
        )

    def create_bucket(self) -> None:
        """
        Create an S3 bucket if it does not already exist.
        
        Raises:
            ClientError: If there's an error creating the bucket.
        """
        try:
            if AWS_REG == "us-east-1":
                self.s3_client.create_bucket(Bucket=self.bucket_name)
            else:
                self.s3_client.create_bucket(
                    Bucket=self.bucket_name,
                    CreateBucketConfiguration={"LocationConstraint": AWS_REG}
                )
            print(f"Bucket '{self.bucket_name}' created successfully.")
        except ClientError as e:
            if e.response['Error']['Code'] == 'BucketAlreadyOwnedByYou':
                print(f"Bucket '{self.bucket_name}' already exists.")
            else:
                print("Error:", e)

    def create_folder(self, folder_name: str) -> None:
        """
        Create a folder in the specified S3 bucket.
        
        Args:
            folder_name (str): The name of the folder to create.
        """
        folder_key = f"{folder_name}/"
        self.s3_client.put_object(Bucket=self.bucket_name, Key=folder_key)
        print(f"Folder '{folder_name}' created in bucket '{self.bucket_name}'.")

    def upload_file(self, folder_name: str, file_path: str) -> None:
        """
        Upload a file to the specified S3 bucket and folder.
        
        Args:
            folder_name (str): The name of the folder to upload the file to.
            file_path (str): The local path of the file to upload.
        
        Raises:
            ClientError: If there's an error uploading the file.
        """
        file_name = os.path.basename(file_path)
        s3_key = f"{folder_name}/{file_name}"
        
        try:
            self.s3_client.upload_file(file_path, self.bucket_name, s3_key)
            print(f"File '{file_name}' uploaded to '{self.bucket_name}/{s3_key}'.")
        except ClientError as e:
            print("Error:", e)

    def upload_annotation_data(self, folder_name: str, annotations: dict) -> None:
        """
        Upload annotation data in JSON format to the specified S3 folder.
        
        Args:
            folder_name (str): The folder in the S3 bucket.
            annotations (dict): The annotation data to be uploaded.
        
        Raises:
            ClientError: If there's an error uploading the annotation data.
        """
        json_data = json.dumps(annotations, indent=4)
        s3_key = f"{folder_name}/annotations.json"
        
        try:
            self.s3_client.put_object(Body=json_data, Bucket=self.bucket_name, Key=s3_key)
            print(f"Annotations uploaded to '{self.bucket_name}/{s3_key}'.")
        except ClientError as e:
            print("Error:", e)

    def update_task_queue(self, folder_name: str, task_queue: dict) -> None:
        """
        Update task queue data in JSON format to the specified S3 folder.
        
        Args:
            folder_name (str): The folder in the S3 bucket.
            task_queue (dict): The task queue data to be uploaded.
        
        Raises:
            ClientError: If there's an error updating the task queue.
        """
        json_data = json.dumps(task_queue, indent=4)
        s3_key = f"{folder_name}/task_queue.json"
        
        try:
            self.s3_client.put_object(Body=json_data, Bucket=self.bucket_name, Key=s3_key)
            print(f"Task queue updated in '{self.bucket_name}/{s3_key}'.")
        except ClientError as e:
            print("Error:", e)

