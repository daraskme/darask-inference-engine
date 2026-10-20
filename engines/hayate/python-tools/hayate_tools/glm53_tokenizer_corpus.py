"""Generate an exact local checkpoint BPE and single-user chat-template corpus."""
import argparse
import json
from pathlib import Path
from .glm53_source import SOURCE


def main():
    from transformers import AutoTokenizer
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=SOURCE)
    parser.add_argument('--out', type=Path, default=Path('/tmp/glm53-tokenizer.jsonl'))
    args = parser.parse_args()
    tokenizer = AutoTokenizer.from_pretrained(args.source, local_files_only=True)
    prompts = ["Hello", "日本語で短く説明して。", "I'm we're I'LL ſ", "1234567890 １２３４５６", "e\u0301 café", "🐈‍⬛❤️🇯🇵", "\t  cats\r\n\n", "<|user|>test", "a\u00a0b\u2003c", "\u0301abc العربية 한글 ①²", "  ", "foo_bar!!\n", "[gMASK]<sop>"]
    with args.out.open('w') as out:
        for prompt in prompts:
            ids = tokenizer.encode(prompt, add_special_tokens=False)
            chat = tokenizer.apply_chat_template([{'role':'user','content':prompt}], tokenize=True, add_generation_prompt=True, return_dict=False)
            out.write(json.dumps({'prompt':prompt, 'ids':ids, 'chat':chat, 'decoded':tokenizer.decode(ids, skip_special_tokens=True)}, ensure_ascii=False)+'\n')
    print(f'{len(prompts)} cases: {args.out}')


if __name__ == '__main__':
    main()
