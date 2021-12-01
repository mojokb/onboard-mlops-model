import datetime
import json
import os
import requests
from minio import Minio
from minio.deleteobjects import DeleteObject
from tqdm import tqdm


class UploadClient:
    def __init__(self, bucket_name, push_event_url):
        self.client = None
        self.bucket_name = bucket_name
        self.image_path = None
        self._set_minio_client()
        self.push_event_url = push_event_url

    def _set_minio_client(self):
        """
        TODO replace key to env
        :return:
        """
        self.client = Minio(
            "192.168.64.5:32008",
            access_key="aimmo_access_key",
            secret_key="aimmo_secret_key",
            region="my-region",
            secure=False
        )

    def upload(self, image_path):
        now = datetime.datetime.now()
        object_prefix = now.strftime("%Y%m%d-%H%M%S")
        print(f"object_prefix {object_prefix}")
        file_list = os.listdir(image_path)

        for file_name in tqdm(file_list):
            self.client.fput_object(
                bucket_name=self.bucket_name,
                object_name=f"{object_prefix}/{file_name}",
                file_path=os.path.join(image_path, file_name))
        data = {'bucket_name': self.bucket_name, 'object_prefix': f'{object_prefix}/'}
        headers = {'Content-Type': 'application/json; charset=utf-8'}
        res = requests.post(self.push_event_url, headers=headers, data=json.dumps(data))
        print(res)

    def delete_objects(self, prefix=None):
        delete_object_list = map(
            lambda x: DeleteObject(x.object_name),
            self.client.list_objects(self.bucket_name, prefix=prefix, recursive=True),
        )
        errors = self.client.remove_objects(self.bucket_name, delete_object_list)
        for error in errors:
            print("error occured when deleting object", error)

if __name__ == "__main__":
    client = UploadClient(bucket_name="torch-raw-images", 
                          push_event_url="http://192.168.64.5:32252/example")
    client.upload("/Users/kb/PycharmProjects/labelers_sdk_test/download/train")