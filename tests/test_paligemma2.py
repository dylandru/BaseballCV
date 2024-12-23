from scripts import LoadTools
from scripts import PaliGemma2

def test_paligemma2_training():
    load_tools = LoadTools()
    test_dataset = load_tools.load_dataset("datasets/yolo/test_dataset.txt")

    classes = {0: 'glove', 1: 'homeplate', 2: 'baseball', 3: 'rubber'}
    model = PaliGemma2(batch_size=1, device = "cpu")
    try:
        model.train(
            dataset=test_dataset,
            epochs=1,
            classes=classes
        )
    except Exception as e:
        print(f"An error occurred: {e}")
        raise