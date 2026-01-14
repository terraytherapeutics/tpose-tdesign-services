"""
S3 client utility for TPose services.
"""

import boto3
import logging
from typing import Optional, Dict, Any, List
from botocore.exceptions import ClientError, NoCredentialsError
import os
import json

logger = logging.getLogger(__name__)


class S3Client:
    """S3 client wrapper with error handling and utilities."""

    def __init__(self, aws_access_key_id: Optional[str] = None,
                 aws_secret_access_key: Optional[str] = None,
                 region_name: str = 'us-west-2'):
        """
        Initialize S3 client.

        Args:
            aws_access_key_id: AWS access key (optional, will use env vars or IAM role)
            aws_secret_access_key: AWS secret key (optional, will use env vars or IAM role)
            region_name: AWS region
        """
        try:
            if aws_access_key_id and aws_secret_access_key:
                self.s3 = boto3.client(
                    's3',
                    aws_access_key_id=aws_access_key_id,
                    aws_secret_access_key=aws_secret_access_key,
                    region_name=region_name
                )
            else:
                # Use default credential chain (env vars, IAM role, etc.)
                self.s3 = boto3.client('s3', region_name=region_name)

            logger.info("S3 client initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize S3 client: {e}")
            raise

    def upload_file(self, local_path: str, bucket: str, s3_key: str) -> bool:
        """
        Upload file to S3.

        Args:
            local_path: Local file path
            bucket: S3 bucket name
            s3_key: S3 object key

        Returns:
            True if successful, False otherwise
        """
        try:
            self.s3.upload_file(local_path, bucket, s3_key)
            logger.info(f"Successfully uploaded {local_path} to s3://{bucket}/{s3_key}")
            return True
        except Exception as e:
            logger.error(f"Failed to upload {local_path} to S3: {e}")
            return False

    def upload_pdb_file(self, local_path: str, bucket: str, s3_key: str) -> bool:
        """
        Upload PDB file to S3 with appropriate content type.

        Args:
            local_path: Local PDB file path
            bucket: S3 bucket name
            s3_key: S3 object key

        Returns:
            True if successful, False otherwise
        """
        try:
            self.s3.upload_file(
                local_path,
                bucket,
                s3_key,
                ExtraArgs={'ContentType': 'chemical/x-pdb'}
            )
            logger.info(f"Successfully uploaded PDB file {local_path} to s3://{bucket}/{s3_key}")
            return True
        except Exception as e:
            logger.error(f"Failed to upload PDB file {local_path} to S3: {e}")
            return False

    def upload_sdf_file(self, local_path: str, bucket: str, s3_key: str) -> bool:
        """
        Upload SDF file to S3 with appropriate content type.

        Args:
            local_path: Local SDF file path
            bucket: S3 bucket name
            s3_key: S3 object key

        Returns:
            True if successful, False otherwise
        """
        try:
            self.s3.upload_file(
                local_path,
                bucket,
                s3_key,
                ExtraArgs={'ContentType': 'chemical/x-mdl-sdfile'}
            )
            logger.info(f"Successfully uploaded SDF file {local_path} to s3://{bucket}/{s3_key}")
            return True
        except Exception as e:
            logger.error(f"Failed to upload SDF file {local_path} to S3: {e}")
            return False

    def download_file(self, bucket: str, s3_key: str, local_path: str) -> bool:
        """
        Download file from S3.

        Args:
            bucket: S3 bucket name
            s3_key: S3 object key
            local_path: Local file path to save to

        Returns:
            True if successful, False otherwise
        """
        try:
            self.s3.download_file(bucket, s3_key, local_path)
            logger.info(f"Successfully downloaded s3://{bucket}/{s3_key} to {local_path}")
            return True
        except Exception as e:
            logger.error(f"Failed to download s3://{bucket}/{s3_key}: {e}")
            return False

    def upload_json(self, data: Dict[str, Any], bucket: str, s3_key: str) -> bool:
        """
        Upload JSON data to S3.

        Args:
            data: Dictionary to upload as JSON
            bucket: S3 bucket name
            s3_key: S3 object key

        Returns:
            True if successful, False otherwise
        """
        try:
            json_str = json.dumps(data, indent=2)
            self.s3.put_object(
                Body=json_str,
                Bucket=bucket,
                Key=s3_key,
                ContentType='application/json'
            )
            logger.info(f"Successfully uploaded JSON to s3://{bucket}/{s3_key}")
            return True
        except Exception as e:
            logger.error(f"Failed to upload JSON to S3: {e}")
            return False

    def download_json(self, bucket: str, s3_key: str) -> Optional[Dict[str, Any]]:
        """
        Download and parse JSON from S3.

        Args:
            bucket: S3 bucket name
            s3_key: S3 object key

        Returns:
            Parsed JSON data or None if failed
        """
        try:
            response = self.s3.get_object(Bucket=bucket, Key=s3_key)
            json_str = response['Body'].read().decode('utf-8')
            data = json.loads(json_str)
            logger.info(f"Successfully downloaded JSON from s3://{bucket}/{s3_key}")
            return data
        except Exception as e:
            logger.error(f"Failed to download JSON from S3: {e}")
            return None

    def file_exists(self, bucket: str, s3_key: str) -> bool:
        """
        Check if file exists in S3.

        Args:
            bucket: S3 bucket name
            s3_key: S3 object key

        Returns:
            True if file exists, False otherwise
        """
        try:
            self.s3.head_object(Bucket=bucket, Key=s3_key)
            return True
        except ClientError as e:
            if e.response['Error']['Code'] == '404':
                return False
            else:
                logger.error(f"Error checking if file exists: {e}")
                return False

    def list_objects(self, bucket: str, prefix: str = '') -> List[str]:
        """
        List objects in S3 bucket with given prefix.

        Args:
            bucket: S3 bucket name
            prefix: Object key prefix

        Returns:
            List of object keys
        """
        try:
            response = self.s3.list_objects_v2(Bucket=bucket, Prefix=prefix)
            if 'Contents' in response:
                return [obj['Key'] for obj in response['Contents']]
            return []
        except Exception as e:
            logger.error(f"Failed to list objects in S3: {e}")
            return []

    def delete_object(self, bucket: str, s3_key: str) -> bool:
        """
        Delete object from S3.

        Args:
            bucket: S3 bucket name
            s3_key: S3 object key

        Returns:
            True if successful, False otherwise
        """
        try:
            self.s3.delete_object(Bucket=bucket, Key=s3_key)
            logger.info(f"Successfully deleted s3://{bucket}/{s3_key}")
            return True
        except Exception as e:
            logger.error(f"Failed to delete S3 object: {e}")
            return False

    @staticmethod
    def from_env() -> 'S3Client':
        """Create S3Client using environment variables for credentials."""
        return S3Client(
            aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID'),
            aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY'),
            region_name=os.getenv('AWS_DEFAULT_REGION', 'us-west-2')
        )
