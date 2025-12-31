 t1 = time.time()
        edge_time = t1 - t0
        audio, sr = librosa.load(edge_output_filename, sr=16000, mono=True)
        duration = len(audio) / sr
        print(f"Audio duration: {duration}s")
        # No duration limitation - removed for better usability
        # if limitation and duration >= max_duration:
        #     print("Error: Audio too long")
        #     return (
        #         f"Audio should be less than {max_duration} seconds in this huggingface space, but got {duration:.2f}s.",
        #         edge_output_filename,
        #         None,
        #     )
