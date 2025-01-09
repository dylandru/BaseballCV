import os
import glob
import shutil
from scripts import LoadTools
from scripts import PaliGemma2


def test_paligemma2_training():
    load_tools = LoadTools()
    test_dataset = load_tools.load_dataset("baseball")

    assert os.path.exists(test_dataset), "Test dataset not found"
    
    mini_test_dir = os.path.join(os.path.dirname(test_dataset), "temp_test_data")
    os.makedirs(mini_test_dir, exist_ok=True)
    
    for ext in ['.jpg', '.txt']:
        files = sorted(glob.glob(os.path.join(test_dataset, f"*{ext}")))[:5]
        for f in files:
            shutil.copy(f, mini_test_dir)
            
    shutil.rmtree(test_dataset)

    classes = {0: 'glove', 1: 'homeplate', 2: 'baseball', 3: 'rubber'}

    try:
        model = PaliGemma2(batch_size=1, device = "cpu")
        model.train(
            dataset=mini_test_dir,
            epochs=1,
            classes=classes
        )
    except Exception as e:
        print(f"An error occurred: {e}")
        raise