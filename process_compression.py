from pydub import AudioSegment
import os
import argparse
import sys
from tqdm import tqdm
from multiprocessing import Pool


def process_file(file_details):
    part, file, bitrate, compression_method, codec, part_type = file_details

    if part_type == 2:
        orig = str(part + "/wav/" + file)
    else:
        orig = str(part + "/flac/" + file)

    pth = str(
        part + "/" + compression_method + "/" + bitrate[:-1] + "/" + file[:-4] + ".flac"
    )
    junk_path = os.path.join(
        "./datasets/LA/data_compression/temp_"
        + file[:-4]
        + "."
        + compression_method.lower()
    )

    try:
        if codec not in ["opus", "ogg", "m4a"]:
            if part_type == 2:
                sig = AudioSegment.from_file(orig, "wav")
            else:
                sig = AudioSegment.from_file(orig, "flac")

            sig.export(junk_path, format=compression_method.lower(), bitrate=bitrate)
            sigNew = AudioSegment.from_file(junk_path, compression_method.lower())
            sigNew.export(pth, format="flac")

        elif codec == "m4a":
            if part_type == 2:
                sig = AudioSegment.from_file(orig, "wav")
            else:
                sig = AudioSegment.from_file(orig, "flac")

            sig.export(junk_path, format="ipod", bitrate=bitrate)
            sigNew = AudioSegment.from_file(junk_path, compression_method.lower())
            sigNew.export(pth, format="flac")

        else:
            os.system(f"ffmpeg -i {orig} -b:a {bitrate} {junk_path} -loglevel 0 -y")
            os.system(
                f"ffmpeg -i {junk_path} -ar 16000 -sample_fmt s16 {pth} -loglevel 0 -y"
            )

    except Exception as e:
        print(f"An error occurred: {e}")

    if os.path.exists(junk_path):
        os.remove(junk_path)


def main():
    # Your existing argument parsing and setup code
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--part", type=int, help="0 = train, 1 = dev, 2 = voc", default=1
    )
    parser.add_argument("--bitrate", type=int, help="0 = 16, 1 = 32, 2 = 64", default=2)
    parser.add_argument("--codec", default="MP3", choices=["MP3", "M4A", "MP4"])
    args = parser.parse_args()

    # Hyper parameters - paths
    train_path = os.path.join("./datasets/LA/ASVspoof2019_LA_train")
    dev_path = os.path.join("./datasets/LA/ASVspoof2019_LA_dev")

    paths_list_big = [train_path, dev_path]
    compression_method = args.codec
    bitrates_big = ["8K", "16K", "32K", "48K", "64K", "128K"]

    paths_list = [paths_list_big[args.part]]
    bitrates = [bitrates_big[args.bitrate]]

    print("Part will be part: {}".format(paths_list))
    print("Bitrate will be will be part: {}".format(bitrates))
    print("Compression Method: {}".format(compression_method))

    # specify codec
    if compression_method == "MP4":
        codec = "aac"
    elif compression_method == "OPUS":
        codec = "opus"
    elif compression_method == "OGG":
        codec = "ogg"
    elif compression_method == "MP3":
        codec = "mp3"
    elif compression_method == "M4A":
        codec = "m4a"

    part = paths_list[0]
    part_type = args.part

    if part_type == 2:
        files = os.listdir(part + "/wav/")
    else:
        files = os.listdir(part + "/flac/")

    bitrate = bitrates[0]
    files_to_process = [
        (part, file, bitrate, compression_method, codec, part_type) for file in files
    ]

    with Pool() as pool:
        list(
            tqdm(pool.imap(process_file, files_to_process), total=len(files_to_process))
        )

    print("Done")


if __name__ == "__main__":
    main()
