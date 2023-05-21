"""Audio Recording Class Example"""
import os
import time
import datetime
import argparse
from queue import Queue
from threading import Thread
from typing import Optional
import pyaudio
import wave
from inputimeout import inputimeout
import warnings
import numpy as np

# Ignore warnings DeprecationWarning
warnings.filterwarnings("ignore", category=DeprecationWarning)
import audioop


def clear_console() -> None:
    """Clears the console depending on the operating system."""
    if os.name == 'posix':
        os.system('clear')  # Mac and Linux
    else:
        os.system('cls')  # Windows


def output_message(message: Optional[str] = None, cls=False) -> None:
    """
    Prints a message with the current time in the format '[YYYY-MM-DD HH:MM:SS] message'.

    :param cls:
    :param message: Message to display.
    """
    if cls:
        clear_console()
    if not message:
        print()
        return
    current_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    output_entry = f"[{current_time}] {message}"
    print(output_entry)


def request_input(message: str, timeout: Optional[int] = 999) -> bool:
    """
    Waits for user input for a specified amount of time and then continues program execution.

    :param message: Message to display before waiting for input.
    :param timeout: Time to wait for input in seconds (if None, then no limit).
    """
    current_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    output_entry = f"[{current_time}] {message}>> "
    try:
        inputimeout(prompt=output_entry, timeout=timeout)
        return True
    except KeyboardInterrupt:
        exit()
    except Exception:
        return False


class AudioProcessor:
    def __init__(self, output_file='recording.wav', min_duration=5, max_duration=10, f_min=20, f_max=50,
                 do_audio_filter="off"):
        """
        Initializes the AudioProcessor object with the specified parameters.

        Args:
        - output_file (str): Name of the output file.
        - min_duration (int): Minimum duration (in seconds) of the recording.
        - max_duration (int): Maximum duration (in seconds) of the recording.
        """
        self.out_stream = None
        self.p = None
        self.in_stream = None
        self.audio_data_queue = None
        self.start_time = None
        self.duration = 0
        self.min_duration = min_duration
        self.max_duration = max_duration
        self.output_file = output_file
        self.f_min = f_min
        self.f_max = f_max
        self.do_audio_filter = do_audio_filter == "on"

        # Create threads for recording audio, playing audio, and timing the recording
        self.process_audio_thread = Thread(target=self.process_audio)
        self.process_audio_thread.daemon = True
        self.play_audio_thread = Thread(target=self.play_audio)
        self.play_audio_thread.daemon = True
        self.timer_thread = Thread(target=self.recording_timer)
        self.timer_thread.daemon = True
        self.sample_rate = 44100

        # Set flags for recording status
        self.is_running = False
        self.maximum_duration_reached = False
        self.minimal_duration_recorded = False
        self.stop_requested = False

    def start(self):
        """
        Starts the audio recording process.
        """
        # Reset variables
        self.p = pyaudio.PyAudio()
        self.in_stream = self.p.open(format=pyaudio.paInt16, channels=1, rate=self.sample_rate, input=True,
                                     frames_per_buffer=1024)
        self.out_stream = self.p.open(format=pyaudio.paInt16, channels=2, rate=self.sample_rate, output=True,
                                      frames_per_buffer=1024)
        self.audio_data_queue = Queue()
        self.maximum_duration_reached = False
        self.minimal_duration_recorded = False
        self.stop_requested = False
        self.start_time = time.time()
        self.duration = 0
        self.is_running = True
        try:
            # Start threads for recording audio, playing audio, and timing the recording
            self.process_audio_thread.start()
            self.play_audio_thread.start()
            self.timer_thread.start()
        except Exception as e:
            print(f"Error starting audio recording: {e}")
            return

    def recording_timer(self):
        """
        Keeps track of the recording time and stops the recording when the maximum duration has been reached.
        """
        try:
            while self.is_running:
                self.duration = round(time.time() - self.start_time)
                time.sleep(1)

                # Minimum recording time control
                if self.duration >= self.min_duration:
                    self.minimal_duration_recorded = True
                    if self.stop_requested:
                        break

                        # Control of limiting the maximum record size
                if self.duration >= self.max_duration:
                    self.maximum_duration_reached = True
                    break
            self.stop()
        except Exception as e:
            print(f"Error in timer: {e}")
            self.stop()

    def stop(self):
        """
        Stops the recording process and cleans up resources.
        """

        if not self.is_running:
            return
        if not self.minimal_duration_recorded:
            self.stop_requested = True
            return
        self.is_running = False
        self.in_stream.stop_stream()
        self.in_stream.close()
        self.in_stream = None
        self.out_stream.stop_stream()
        self.out_stream.close()
        self.out_stream = None
        self.audio_data_queue = None
        self.p.terminate()
        self.p = None

    def filter_by_frequency_range(self, data, lowpass, highpass):
        try:
            da = np.fromstring(data, dtype=np.int16)
            left, right = da[0::2], da[1::2]  # left and right channel
            lf, rf = np.fft.rfft(left), np.fft.rfft(right)
            lf[:lowpass], rf[:lowpass] = 0, 0  # low pass filter
            # lf[55:66], rf[55:66] = 0, 0 # line noise
            lf[highpass:], rf[highpass:] = 0, 0  # high pass filter
            nl, nr = np.fft.irfft(lf), np.fft.irfft(rf)
            ns = np.column_stack((nl, nr)).ravel().astype(np.int16)
            return ns.tostring()
        except Exception as e:
            # If an exception occurs, print an error message and stop the recording process
            print(f"Error filter audio: {e}")
            self.stop()

    def process_audio(self):
        while self.is_running:
            try:
                # Read 1024 bytes of audio data from input stream
                audio_data = self.in_stream.read(1024)
                # Filtering frequency range
                if self.do_audio_filter:
                    audio_data = self.filter_by_frequency_range(audio_data, self.f_min, self.f_max)
                audio_data = audioop.tostereo(audio_data, 2, 2, 2)
                # Double the volume of the audio data using audio.mul
                audio_data = audioop.mul(audio_data, 2, 2)
                # Put the modified audio data in the audio data queue
                self.audio_data_queue.put(audio_data)
            except Exception as e:
                # If an exception occurs, print an error message and stop the recording process
                print(f"Error recording audio: {e}")
                self.stop()

    def play_audio(self):
        # Open a new wave file for writing
        wav_file = wave.open(self.output_file, 'wb')
        try:
            # Set the wave file parameters
            wav_file.setnchannels(2)
            wav_file.setsampwidth(self.p.get_sample_size(pyaudio.paInt16))
            wav_file.setframerate(44100)
            while self.is_running:
                try:
                    # Get modified audio data from the audio data queue
                    audio_data = self.audio_data_queue.get()
                    # Play the modified audio data using the output stream
                    self.out_stream.write(audio_data)
                    # Write the modified audio data to the wave file
                    wav_file.writeframes(audio_data)
                except Exception as e:
                    # If an exception occurs, print an error message and stop the recording process
                    print(f"Error playing audio: {e}")
                    self.stop()
            # Close the wave file after the recording process is stopped
            wav_file.close()
        except Exception as e:
            # If an exception occurs, print an error message and stop the recording process
            print(f"Error opening file to record audio: {e}")
            self.stop()


if __name__ == '__main__':

    # parse command line arguments
    parser = argparse.ArgumentParser(description='Audio recorder')
    parser.add_argument('-p', '--path', help='File path for audio recording', default='recording.wav')
    parser.add_argument('-n', '--min', help='Minimum duration (in seconds) of the recording', default=5)
    parser.add_argument('-x', '--max', help='Maximum duration (in seconds) of the recording', default=60)
    parser.add_argument('-lo', '--low', help='Low frequency filter', default=21)
    parser.add_argument('-hi', '--high', help='High frequency filter', default=9000)
    parser.add_argument('-f', '--filter', help='Frequency Filter on/off', default='off')
    args = parser.parse_args()
    # prompt user to start audio recording

    while True:
        output_message(f"Wellcome to audio recorder!", True)
        output_message(f"")
        output_message(f"Output Path/Filename: '{args.path}'")
        output_message(f"Minimum recording time: {args.min} seconds")
        output_message(f"Maximum recording time: {args.max} seconds")
        output_message(f"Low Frequency: {args.low} seconds")
        output_message(f"High Frequency: {args.high} seconds")
        output_message(f"Frequency Filter: {args.filter}")
        output_message(f"")
        request_input(f"Press 'Enter' to Start recording...")

        # create an instance of the AudioProcessor class and start recording
        audio_processor = AudioProcessor(args.path, int(args.min), int(args.max), int(args.low), int(args.high),
                                         args.filter)
        audio_processor.start()

        while audio_processor.is_running:
            # prompt user to stop audio recording
            # display the minimum and maximum recording duration parameters
            output_message("Press 'Enter' to stop audio recording", True)
            if request_input(f"Recording: {audio_processor.duration} sec...", 1):
                audio_processor.stop()
                break

        if not audio_processor.minimal_duration_recorded:
            while not audio_processor.minimal_duration_recorded:
                output_message(f"Finalizing... Minimal recording time: {audio_processor.min_duration} sec.", True)
                seconds_left = audio_processor.min_duration - audio_processor.duration
                request_input(f"Please wait: {seconds_left} sec...", 1)

        audio_processor.stop()

        clear_console()
        if audio_processor.maximum_duration_reached:
            output_message(f"Maximum recording duration reached: {audio_processor.max_duration} sec...")
        output_message("Audio recording stopped...")
        output_message(f"Output file - '{audio_processor.output_file}', duration: {audio_processor.duration} sec.")
        # prompt user to start audio recording
        request_input("Press 'Enter' to Restart audio recording")

    time.sleep(1)
    sys.exit