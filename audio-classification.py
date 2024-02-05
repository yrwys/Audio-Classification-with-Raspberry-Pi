"""Main scripts to run audio classification."""
import argparse
import logging
import time
import numpy as np
from scipy.fft import fft
from tflite_support.task import audio
from tflite_support.task import core
from tflite_support.task import processor
from utils import Plotter

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def run(model: str, max_results: int, score_threshold: float,
        overlapping_factor: float, num_threads: int,
        enable_edgetpu: bool) -> None:
    """Continuously run inference on audio data acquired from the device.

    Args:
        model: Name of the TFLite audio classification model.
        max_results: Maximum number of classification results to display.
        score_threshold: The score threshold of classification results.
        overlapping_factor: Target overlapping between adjacent inferences.
        num_threads: Number of CPU threads to run the model.
        enable_edgetpu: Whether to run the model on EdgeTPU.
    """
    if not (0 < overlapping_factor < 1.0):
        raise ValueError('Overlapping factor must be between 0 and 1.')

    if not (0 <= score_threshold <= 1.0):
        raise ValueError('Score threshold must be between (inclusive) 0 and 1.')

    # Initialize the audio classification model.
    base_options = core.BaseOptions(
        file_name=model, use_coral=enable_edgetpu, num_threads=num_threads)
    classification_options = processor.ClassificationOptions(
        max_results=max_results, score_threshold=score_threshold)
    options = audio.AudioClassifierOptions(
        base_options=base_options, classification_options=classification_options)
    classifier = audio.AudioClassifier.create_from_options(options)

    # Initialize the audio recorder and a tensor to store the audio input.
    audio_record = classifier.create_audio_record()
    tensor_audio = classifier.create_input_tensor_audio()

    # We'll try to run inference every interval_between_inference seconds.
    # This is usually half of the model's input length to create overlapping
    # between incoming audio segments to improve classification accuracy.
    input_length_in_second = float(len(
        tensor_audio.buffer)) / tensor_audio.format.sample_rate
    interval_between_inference = input_length_in_second * (1 - overlapping_factor)
    pause_time = interval_between_inference * 0.1
    last_inference_time = time.time()

    # Initialize a plotter instance to display the classification results.
    plotter = Plotter()

    # Start audio recording in the background.
    audio_record.start_recording()

    # Loop until the user closes the classification results plot.
    while True:
        now = time.time()
        diff = now - last_inference_time
        if diff < interval_between_inference:
            time.sleep(pause_time)
            continue
        last_inference_time = now

        try:
            tensor_audio.load_from_audio_record(audio_record)
            result = classifier.classify(tensor_audio)

            # Obtain time and frequency domain data
            time_domain = tensor_audio.buffer
            frequency_domain = np.abs(fft(time_domain))

            # Plot the classification results, time domain, and frequency domain
            plotter.plot(result, time_domain, frequency_domain)

        except Exception as e:
            logger.error(f"Error during classification: {e}")

def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        '--model',
        help='Name of the audio classification model.',
        required=False,
        default='tflite/yamnet.tflite')
    parser.add_argument(
        '--max_results',
        type=int,
        help='Maximum number of results to show.',
        required=False,
        default=3)
    parser.add_argument(
        '--overlapping_factor',
        type=float,
        help='Target overlapping between adjacent inferences. Value must be in (0, 1)',
        required=False,
        default=0.5)
    parser.add_argument(
        '--score_threshold',
        type=float,
        help='The score threshold of classification results.',
        required=False,
        default=0.0)
    parser.add_argument(
        '--num_threads',
        type=int,
        help='Number of CPU threads to run the model.',
        required=False,
        default=4)
    parser.add_argument(
        '--enable_edge_tpu',
        help='Whether to run the model on EdgeTPU.',
        action='store_true',
        required=False,
        default=False)
    args = parser.parse_args()

    try:
        run(args.model, args.max_results, args.score_threshold,
            args.overlapping_factor, args.num_threads, args.enable_edge_tpu)
    except KeyboardInterrupt:
        logger.info("Classification interrupted by user.")

if __name__ == '__main__':
    main()