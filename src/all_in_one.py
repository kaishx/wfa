import os
import shutil
import sys

target_dir = [
    # you should change this if you place files in another folder or move them. i think this accepts >1 directories if you hardcode it.
    os.path.join(os.getcwd(), 'wfa_logs')
]


class LogProcessorPipeline:
    def __init__(self, base_dir, pairs_list, lines_to_read=15):
        self.base_dir = base_dir
        self.pairs = pairs_list
        self.lines_to_read = lines_to_read
        self.output_dir = "wfa_outputs"  # used only if you want to separate outputs, usually ignored for this logic

    def organize(self):
        print("1st part: organizing...")

        moved_count = 0

        files = [f for f in os.listdir(self.base_dir) if os.path.isfile(os.path.join(self.base_dir, f))]

        for filename in files:
            if not filename.endswith(('.log', '.txt', '.csv')):
                continue

            # assumes the filename is asset1_asset2_... or else on god it breaks
            matched_pair = None

            for pair in self.pairs:
                pair_filename_format = pair.replace(" ", "_")

                if filename.startswith(pair_filename_format):
                    matched_pair = pair
                    break

            if matched_pair:
                folder_name = f"#{matched_pair}"
                target_folder = os.path.join(self.base_dir, folder_name)

                os.makedirs(target_folder, exist_ok=True)

                src_path = os.path.join(self.base_dir, filename)
                dst_path = os.path.join(target_folder, filename)

                try:
                    shutil.move(src_path, dst_path)
                    print(f"moved: {filename} to {folder_name}")
                    moved_count += 1
                except Exception as e:
                    print(f"!!! something happened while trying to move {filename}: {e}!!!")

        print(f"1st part complete, moved {moved_count} files.\n")

    def summarize(self):
        print(f"2nd part: summarizing the last {self.lines_to_read} lines of each raw data in a file")

        for pair in self.pairs:
            folder_name = f"#{pair}"
            folder_path = os.path.join(self.base_dir, folder_name)

            if not os.path.exists(folder_path):
                continue

            output_filename = os.path.join(folder_path, f"compiled_#{pair}.txt")
            compiled_content = []
            file_count = 0

            for item in os.listdir(folder_path):
                file_path = os.path.join(folder_path, item)

                if (os.path.isfile(file_path)
                        and item.endswith(('.log', '.txt'))
                        and item != os.path.basename(output_filename)
                        and not item.startswith("compiled_")):

                    file_count += 1
                    try:
                        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                            lines = f.readlines()
                            last_lines = lines[-self.lines_to_read:]

                            compiled_content.append("=" * 60)
                            compiled_content.append(f"FILE: {item}")
                            compiled_content.append("=" * 60)
                            compiled_content.extend([line.strip() for line in last_lines if line.strip()])
                            compiled_content.append("\n\n")
                    except Exception as e:
                        print(f"!!! error reading {item}: {e} !!!")

            if compiled_content:
                try:
                    with open(output_filename, 'w', encoding='utf-8') as outfile:
                        outfile.write('\n'.join(compiled_content))
                    print(f"Compiled {file_count} logs for {pair}")
                except Exception as e:
                    print(f"!!! error writing output for {pair}: {e} !!!")
            else:
                print(f"!!! no logs found for {pair} !!!")

        print(f"2nd part complete\n")

    def compile(self):
        print("=" * 50)
        print("3rd part: compiling the summarized ones into one highest order file")
        print("=" * 50)

        master_output_path = os.path.join(self.base_dir, 'master_compiled_analysis.txt')
        master_content = []
        pairs_processed = 0

        for pair in self.pairs:
            folder_name = f"#{pair}"
            folder_path = os.path.join(self.base_dir, folder_name)
            compiled_filename = f"compiled_#{pair}.txt"
            compiled_file_path = os.path.join(folder_path, compiled_filename)

            if not os.path.exists(compiled_file_path):
                continue

            try:
                with open(compiled_file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()

                master_content.append("\n" * 2 + "#" * 80)
                master_content.append(f"MASTER BLOCK START: {pair}")
                master_content.append("#" * 80 + "\n")
                master_content.append(content)
                pairs_processed += 1
            except Exception as e:
                print(f"error reading summary for {pair}: {e}")

        if pairs_processed > 0:
            try:
                with open(master_output_path, 'w', encoding='utf-8') as outfile:
                    outfile.write('\n'.join(master_content))
                print(f"Master file created: {master_output_path}")
                print(f"Merged data from {pairs_processed} asset pairs.")
            except Exception as e:
                print(f"!!! error writing master file: {e}!!!")
        else:
            print(" !!!No compiled pair data found to merge !!!")

        print(f"3rd part complete.\n")

    def run(self):
        print(f"starting all in one for: {self.base_dir}")
        self.organize()
        self.summarize()
        self.compile()
        print(f"{self.base_dir} done compiling")


if __name__ == "__main__":
    PAIRS_CONFIG = [
    "V MA"
    # follow the template "XXX YYY"
    ]

    for log_directory in target_dir:
        if not os.path.exists(log_directory):
            print(f"The directory {log_directory} cannot be found, skipping it")
            continue

        pipeline = LogProcessorPipeline(
            base_dir=log_directory,
            pairs_list=PAIRS_CONFIG,
            lines_to_read=15 # i hardcoded 15, but if the file format of the txt log ever changes, edit it as such ;)
        )
        pipeline.run()

    print("compilation finished successfully")