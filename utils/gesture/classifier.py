from tensorflow import lite
import numpy


class GestureClassifier(object):
    def __init__(
        self,
        model_path=r"D:\Python Projects\Hand Gesture Classification\utils\gesture\models\sample\model.tflite",
        num_threads=1,
    ) -> None:
        """loads .tflite model

        Args:
            model_path (regexp, optional): path to model. Defaults to D:\\Python Projects\\Hand Gesture Classification\\utils\\gesture\\models\\sample\\model.tflite.
            num_threads (int, optional): number of threads. Defaults to 1.
        """
        self.interpreter = lite.Interpreter(
            model_path=model_path, num_threads=num_threads
        )

        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

    def __call__(self, data: numpy.ndarray) -> int:
        """model inference

        Args:
            data (numpy.ndarray): normalised hand landmarks

        Returns:
            int: index
        """
        tensor_index = self.input_details[0]["index"]

        self.interpreter.set_tensor(tensor_index, numpy.float32([data.copy()]))

        self.interpreter.invoke()

        tensor_index = self.output_details[0]["index"]
        
        return numpy.argmax(self.interpreter.get_tensor(tensor_index))
