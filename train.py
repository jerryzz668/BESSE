import argparse
import os
import time
import torch
from torch.nn import DataParallel
from torch.utils.data import DataLoader, Dataset
from transformers import BertTokenizer, EncoderDecoderModel, AdamW
from tqdm import tqdm
from tensorboardX import SummaryWriter  # 导入 TensorBoardX

from dataset import CoupletDataset
from utils import mkdir

# 参数解析器
def parse_args():
    parser = argparse.ArgumentParser(description="Train a BERT model for punctuation and typo correction.")
    parser.add_argument("--train_path", type=str, default='datasets/couplet/train_sources_100', help="Path to the training data.")
    parser.add_argument("--target_path", type=str, default='datasets/couplet/train_targets_100', help="Path to the target data.")
    parser.add_argument("--output_dir", type=str, default="./saved_model/", help="Path to save the trained model.")
    parser.add_argument("--epochs", type=int, default=200, help="Number of training epochs.")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for training.")
    parser.add_argument("--learning_rate", type=float, default=5e-5, help="Learning rate for optimizer.")
    parser.add_argument("--max_len", type=int, default=32, help="Maximum sequence length for BERT model.")
    parser.add_argument("--save_interval", type=int, default=50, help="save_interval.")
    parser.add_argument("--gpus", type=str, default='0', help="k-th gpu.")
    args = parser.parse_args()
    return args

# 主函数
def main():
    args = parse_args()

    current_time = time.strftime("%Y%m%d_%H%M%S", time.localtime())
    mkdir(args.output_dir+'{}'.format(current_time))

    # 设置GPU设备
    os.environ["CUDA_VISIBLE_DEVICES"] = "{}".format(args.gpus)  # args.gpus
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 分词器和数据集
    tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
    couplet_dataset = CoupletDataset(tokenizer, args.train_path, args.target_path, max_len=args.max_len)

    # 数据加载器
    data_loader = DataLoader(couplet_dataset, batch_size=args.batch_size, shuffle=True)

    # 模型
    model = EncoderDecoderModel.from_encoder_decoder_pretrained('bert-base-chinese', 'bert-base-chinese')
    # 添加以下代码以配置decoder_start_token_id
    model.config.decoder_start_token_id = tokenizer.cls_token_id
    # 添加以下代码以配置pad_token_id
    model.config.pad_token_id = tokenizer.pad_token_id

    model.to(device)

    # 优化器
    optimizer = AdamW(model.parameters(), lr=args.learning_rate)

    # TensorBoard
    tensorboard_writer = SummaryWriter(logdir=args.output_dir)  # 指定 TensorBoard 日志目录

    # 训练
    model.train()
    for epoch in range(args.epochs):
        total_loss = 0
        loop = tqdm(data_loader, leave=True)
        for batch in loop:
            optimizer.zero_grad()
            inputs = {key: val.to(device) for key, val in batch.items()}
            outputs = model(**inputs)

            loss = outputs.loss
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            loop.set_description(f'Epoch {epoch+1}/{args.epochs}')
            loop.set_postfix(loss=loss.item())
        loop.close()

        avg_loss = total_loss / len(data_loader)
        print(f"Epoch {epoch+1}/{args.epochs} - Loss: {avg_loss:.4f}")


        # 检查是否达到保存模型的epoch
        if (epoch + 1) % args.save_interval == 0:
            save_path = os.path.join(args.output_dir, f"model_epoch_{epoch+1}.pt")
            # 保存模型时的代码
            model.save_pretrained(save_path)

            print(f"Saved model checkpoint to '{save_path}'")

        # 记录 TensorBoard 日志
        tensorboard_writer.add_scalar('Loss/Train', avg_loss, epoch)

    # 保存最后的模型
    final_save_path = os.path.join(args.output_dir, "final_model.pt")
    # 保存模型时的代码
    model.save_pretrained(final_save_path)

    print(f"Saved final model to '{final_save_path}'")

    # 关闭 TensorBoard
    tensorboard_writer.close()

if __name__ == "__main__":
    main()
