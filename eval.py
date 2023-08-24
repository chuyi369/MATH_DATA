import os, re
import fire
from dataset import read_jsonl, last_boxed_only_string, is_equiv, remove_boxed
def main(data_path: str = ""):
    overall_right=0
    overall_num=0
    all_cate = ["counting_and_probability", "intermediate_algebra", "number_theory", "precalculus", "prealgebra",
                "geometry", "algebra", "500"]
    #all_cate = ["counting_and_probability", "number_theory", "precalculus", "prealgebra",
    #            "geometry", "algebra", "500"]
    for cate in all_cate:
        path = os.path.join(data_path, f"test_{cate}.jsonl")
        if os.path.exists(path):
            correct = 0
            #num = 0
            examples = read_jsonl(path)
            for sample in examples:
                #if len(sample["generated_answer"]) > 5100:
                #    num += 1
                #    continue
                output = remove_boxed(last_boxed_only_string(sample["generated_answer"]))
                label = remove_boxed(last_boxed_only_string(sample["answer"]))
                equiv = is_equiv(output, label)
                if equiv:
                    if cate!="500":
                        overall_right+=1
                    correct += 1
            #print('more than 510', num)
                if cate!="500":
                    overall_num+=1
            print(cate, "test acc: ", 100 * correct / (len(examples)))
    print("overall:{}".format(100*overall_right/(overall_num+1e-8)))

    overall_right=0
    overall_num=0
    for cate in all_cate:
        path = os.path.join(data_path, f"test_{cate}_n16_1.jsonl")
        if os.path.exists(path):
            correct = 0
            #num = 0
            examples = read_jsonl(path)
            for sample in examples:
                #if len(sample["generated_answer"]) > 5100:
                #    num += 1
                #    continue
                for generated_answer in sample["generated_answer"]:
                    output = remove_boxed(last_boxed_only_string(generated_answer))
                    label = remove_boxed(last_boxed_only_string(sample["answer"]))
                    equiv = is_equiv(output, label)
                    if equiv:
                        if cate!="500":
                            overall_right+=1
                        correct += 1
                #print('more than 510', num)
                    if cate!="500":
                        overall_num+=1
            print(cate, "acc: ", 100 * correct / ((len(examples)) * 16))
    print("overall:{}".format(100*overall_right/(overall_num+1e-8)))


    overall_right=0
    overall_num=0
    for cate in all_cate:
        path = os.path.join(data_path, f"train_{cate}_n16_2.jsonl")
        if os.path.exists(path):
            correct = 0
            #num = 0
            examples = read_jsonl(path)
            for sample in examples:
                #if len(sample["generated_answer"]) > 5100:
                #    num += 1
                #    continue
                for generated_answer in sample["generated_answer"]:
                    output = remove_boxed(last_boxed_only_string(generated_answer))
                    label = remove_boxed(last_boxed_only_string(sample["answer"]))
                    equiv = is_equiv(output, label)
                    if equiv:
                        if cate!="500":
                            overall_right+=1
                        correct += 1
                #print('more than 510', num)
                    if cate!="500":
                        overall_num+=1
            print(cate, "acc: ", 100 * correct / ((len(examples)) * 16))
    print("overall:{}".format(100*overall_right/(overall_num+1e-8)))


    path = os.path.join(data_path, "gsm8k_test.jsonl")
    if os.path.exists(path):
        correct = 0
        examples = read_jsonl(path)
        for sample in examples:
            number_list = re.findall(r"\d+\.?\d*", sample["generated_answer"])
            answer_list = re.findall(r"\d+\.?\d*", sample["answer"])
            try:
                final_answer = number_list[-1].strip('.')
            except:
                continue
            label = answer_list[-1].strip('.')
            if label == final_answer:
                correct += 1
        print(data_path, " gsm8k acc: ", 100 * correct / len(examples))
    path = os.path.join(data_path, "asdiv_test.jsonl")
    if os.path.exists(path):
        correct = 0
        examples = read_jsonl(path)
        for sample in examples:
            number_list = re.findall(r"\d+\.?\d*", sample["generated_answer"])
            answer_list = re.findall(r"\d+\.?\d*", sample["answer"])
            try:
                final_answer = number_list[-1].strip('.')
            except:
                continue
            label = answer_list[-1].strip('.')
            if label == final_answer:
                correct += 1
        print(data_path, " asdiv acc: ", 100 * correct / len(examples))
    path = os.path.join(data_path, "svamp_test.jsonl")
    if os.path.exists(path):
        correct = 0
        examples = read_jsonl(path)
        for sample in examples:
            number_list = re.findall(r"\d+\.?\d*", sample["generated_answer"])
            #answer_list = re.findall(r"\d+\.?\d*", sample["answer"])
            label = str(int(sample["answer"]))
            try:
                final_answer = number_list[-1].strip('.')
            except:
                continue
            #label = answer_list[-1].strip('.')
            if label == final_answer:
                correct += 1
        print(data_path, " svamp acc: ", 100 * correct / len(examples))

    path = os.path.join(data_path, "multi_test.jsonl")
    if os.path.exists(path):
        correct = 0
        examples = read_jsonl(path)
        for sample in examples:
            number_list = re.findall(r"\d+\.?\d*", sample["generated_answer"])
            #answer_list = re.findall(r"\d+\.?\d*", sample["answer"])
            label = str(int(sample["answer"]))
            try:
                final_answer = number_list[-1].strip('.')
            except:
                continue
            #label = answer_list[-1].strip('.')
            if label == final_answer:
                correct += 1
        print(data_path, " multi acc: ", 100 * correct / len(examples))


if __name__ == '__main__':
    fire.Fire(main)
