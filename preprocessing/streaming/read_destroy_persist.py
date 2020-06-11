import os
from sparkmodeling.common.lodatastruct import LODataStruct

IN_FOLDER = "output--"
OUT_FOLDER = "output"


def main():
    if not os.path.exists(OUT_FOLDER):
        os.makedirs(OUT_FOLDER)

    for fname in os.listdir(IN_FOLDER):
        if ".bin" in fname:
            fpath = os.path.join(IN_FOLDER, fname)
            lods = LODataStruct.load_from_file(
                os.path.join(fpath), autobuild=False)
            lods.serialize(os.path.join(OUT_FOLDER, fname), destroy=True)


if __name__ == "__main__":
    main()
