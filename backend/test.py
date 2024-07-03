import requests


def test_request(
    url: str,
    input_files: dict,
    payload: dict,
):
    is_stream = False
    if "response_format" in payload:
        is_stream = True if payload["response_format"] == "stream" else False

    response = requests.post(url,
                             data=payload,
                             files=input_files,
                             stream=is_stream)

    if response.status_code != 200:
        return print(response.json())

    return response


if __name__ == "__main__":

    SERVER_URL = "http://localhost:5000"
    INPUT_FILE_PATH = "C:\PATH\TEST_FILE.mp4"

    # Test /transcription
    with open(INPUT_FILE_PATH, "rb") as file:
        files = {"file": file}
        payload = {
            "response_format": "stream",
            "model_size": "large-v2",
            "task": "transcribe",
            "vad_filter": True,
            "is_diarization": True,
        }

        response = test_request(
            url=SERVER_URL + "/transcription",
            input_files=files,
            payload=payload
        )

        if payload["response_format"] == "stream":
            for line in response.iter_lines():
                decoded_line = line.decode('utf-8')
                print(decoded_line)
        elif payload["response_format"] == "json":
            print(response.json())

    # Test /diarization
    with open(INPUT_FILE_PATH, "rb") as file:
        files = {"file": file}
        payload = {
            "use_auth_token": "huggingface_read_token",
        }
        response = test_request(
            url=SERVER_URL + "/diarization",
            input_files=files,
            payload=payload
        )
        print(response.json())

    # Test /vad
    with open(INPUT_FILE_PATH, "rb") as file:
        files = {"file": file}
        payload = {
            "threshold": 0.5
        }
        response = test_request(
            url=SERVER_URL + "/vad",
            input_files=files,
            payload=payload
        )

        output_path = "vad_test.wav"
        with open(output_path, "wb") as vad_output_file:
            vad_output_file.write(response.content)
