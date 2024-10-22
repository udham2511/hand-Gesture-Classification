from tensorflow import lite
import numpy


class HistoryClassifier(object):
    def __init__(
        self,
        model_path=r"D:\Python Projects\Hand Gesture Classification\utils\history\models\sample\model.tflite",
        num_threads=1,
        threshold=0.5,
        invalid_value=0,
    ) -> None:
        """loads .tflite model

        Args:
            model_path (regexp, optional): path to model. Defaults to D:\\Python Projects\\Hand Gesture Classification\\utils\\history\\models\\sample\\model.tflite.
            num_threads (int, optional): number of threads. Defaults to 1.
        """
        self.interpreter = lite.Interpreter(
            model_path=model_path, num_threads=num_threads
        )

        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

        self.threshold = threshold
        self.invalid_value = invalid_value

    def __call__(self, data: numpy.ndarray) -> int:
        """model inference

        Args:
            data (numpy.ndarray): normalised finger landmarks

        Returns:
            int: index
        """
        tensor_index = self.input_details[0]["index"]

        self.interpreter.set_tensor(tensor_index, numpy.float32([data.copy()]))

        self.interpreter.invoke()

        tensor_index = self.output_details[0]["index"]

        result = numpy.squeeze(self.interpreter.get_tensor(tensor_index))

        index = numpy.argmax(result)

        return index if result[index] >= self.threshold else self.invalid_value
