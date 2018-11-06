import argparse

def main(args):
    print('Reading answer_file and output_file')
    print(f'answer_file={args.answer_file}, output_file={args.output_file}')
    with open(args.answer_file, 'r', encoding='utf-8') as answer_file, open(args.output_file, 'r', encoding='utf-8') as output_file:
        total_count = 0
        total_accuracy = 0
        for index, (answer_line, output_line) in enumerate(zip(answer_file, output_file)):
            total_count += 1
            answer_line, output_line = answer_line.strip(), output_line.strip()
            answer_tokens = answer_line.split()
            output_tokens = output_line.split()
            output_que = output_tokens
            answer_length = len(answer_tokens)
            correct_count = 0
            for answer_token in answer_tokens:
                # if not (answer_token in output_tokens):
                #    print(index, (answer_token in output_tokens), answer_token, output_tokens)
                if answer_token in output_que:
                    correct_count += 1
                    output_que = output_que[output_que.index(answer_token) + 1:]
            single_accuracy = correct_count / answer_length
            total_accuracy = ((total_accuracy * index) + single_accuracy) / (index + 1)
            if index % 10000 == 0:
                print_dic({'index': index, 'answer_length': answer_length, 'total_acc': total_accuracy,
                           'single_acc': single_accuracy, 'answer': answer_tokens, 'output': output_tokens})
        print('TEST END ------------------------------------------')
        print(f'total_accuracy={total_accuracy}, total_number={index + 1}')
    return


def print_dic(dic):
    print(','.join([f'{k}={v}' for k, v in dic.items()]))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_file', help='Location of output(prediction) file')
    parser.add_argument('--answer_file', help='Location of answer(target) file')
    args = parser.parse_args()
    main(args)

