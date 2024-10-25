from modules.translation.deepl_api import DeepLAPI
from modules.translation.nllb_inference import NLLBInference
from test_config import *

import os
import pytest


@pytest.mark.parametrize("model_size, file_path", [
    (TEST_NLLB_MODEL, TEST_SUBTITLE_SRT_PATH),
    (TEST_NLLB_MODEL, TEST_SUBTITLE_VTT_PATH),
])
def test_nllb_inference(
    model_size: str,
    file_path: str
):
    nllb_inferencer = NLLBInference()
    print(f"NLLB Device : {nllb_inferencer.device}")

    result_str, file_paths = nllb_inferencer.translate_file(
        fileobjs=[file_path],
        model_size=model_size,
        src_lang="eng_Latn",
        tgt_lang="kor_Hang",
    )

    assert isinstance(result_str, str)
    assert isinstance(file_paths[0], str)


@pytest.mark.skipif(
    os.getenv("DEEPL_API_KEY") is None or not os.getenv("DEEPL_API_KEY"),
    reason="DeepL API key is unavailable"
)
@pytest.mark.parametrize("file_path", [
    TEST_SUBTITLE_SRT_PATH,
    TEST_SUBTITLE_VTT_PATH,
])
def test_deepl_api(
    file_path: str
):
    deepl_api = DeepLAPI()

    api_key = os.getenv("DEEPL_API_KEY")

    result_str, file_paths = deepl_api.translate_deepl(
        auth_key=api_key,
        fileobjs=[file_path],
        source_lang="English",
        target_lang="Korean",
        is_pro=False,
        add_timestamp=True,
    )

    assert isinstance(result_str, str)
    assert isinstance(file_paths[0], str)
