import utils
from data.load_data import load_data_from_json
from models.baseline import bart

if __name__ == "__main__":
    args = utils.get_args()
    
    data_list = load_data_from_json(args.data_path, args.data_option, args.key_option)
    if args.model == 'bart':
        output, gold_metareview = bart.run_bart_summarization(args.sample_size, data_list)
    
    if output is None or gold_metareview is None:
        print("No output or gold meta-review generated.")
        exit(1)
    
    with open(args.output_path, 'w') as f:
        for i in range(len(output)):
            f.write(f"Generated Summary {i+1}:\n")
            f.write(output[i] + "\n")
            f.write(f"Gold Metareview {i+1}:\n")
            f.write(gold_metareview[i] + "\n")
            f.write("\n")
    print(f"Summarization output saved to {args.output_path}")    
