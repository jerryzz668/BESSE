import torch, os
from transformers import BertTokenizer, EncoderDecoderModel

def inference(model, tokenizer, input_text, max_length=50):
    model.eval()
    input_ids = tokenizer.encode(input_text, return_tensors="pt")
    input_ids = input_ids.to(model.device)

    # 生成输出序列
    with torch.no_grad():
        output_ids = model.generate(input_ids, max_length=max_length)

    # 解码生成的序列
    output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    return output_text

def main():
    # 加载训练好的模型和分词器
    model_path = "/home/jerry/Documents/code/general_seq2seq/model_save/model_epoch_200.pt"  # 替换为你的训练模型的路径
    model = EncoderDecoderModel.from_pretrained(model_path)
    tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')

    # 设置GPU设备
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # args.gpus
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    # 输入文本
    input_text = "凹凹凸凸凹凸不平"

    # 进行推理
    output_text = inference(model, tokenizer, input_text)

    # 打印输出
    print(f"Input: {input_text}")
    print(f"Output: {output_text}")

if __name__ == "__main__":
    main()
