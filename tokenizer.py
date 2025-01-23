import sacremoses
import subword_nmt.learn_bpe
import subword_nmt.apply_bpe

class ParallelCorpusTokenizer:
    def __init__(self):
        self.en_tokenizer = sacremoses.MosesTokenizer(lang='en')
        self.de_tokenizer = sacremoses.MosesTokenizer(lang='de')
    
    def tokenize_files(self, en_file_path, de_file_path, output_en_path, output_de_path):
        """
        Tokenize parallel corpus files
        
        Args:
            en_file_path (str): Path to English sentences file
            de_file_path (str): Path to German sentences file
            output_en_path (str): Path to save tokenized English sentences
            output_de_path (str): Path to save tokenized German sentences
        """
        with open(en_file_path, 'r', encoding='utf-8') as en_file, \
             open(de_file_path, 'r', encoding='utf-8') as de_file, \
             open(output_en_path, 'w', encoding='utf-8') as out_en, \
             open(output_de_path, 'w', encoding='utf-8') as out_de:
            
            for en_line, de_line in zip(en_file, de_file):
                # Tokenize and write English sentences
                en_tokens = self.en_tokenizer.tokenize(en_line.strip())
                out_en.write(' '.join(en_tokens) + '\n')
                
                # Tokenize and write German sentences
                de_tokens = self.de_tokenizer.tokenize(de_line.strip())
                out_de.write(' '.join(de_tokens) + '\n')
    
    def learn_bpe(self, input_file, output_codes_path, num_symbols=10000):
        """Learn BPE codes from input file"""
        with open(input_file, 'r', encoding='utf-8') as infile, \
             open(output_codes_path, 'w', encoding='utf-8') as outfile:
            subword_nmt.learn_bpe.learn_bpe(
                infile, 
                outfile, 
                num_symbols=num_symbols
            )
    
    def apply_bpe(self, input_path, output_path, codes_path):
        """Apply BPE to tokenized corpus"""
        with open(codes_path, 'r', encoding='utf-8') as codes_file:
            bpe = subword_nmt.apply_bpe.BPE(codes_file)
        
        with open(input_path, 'r', encoding='utf-8') as input_file, \
             open(output_path, 'w', encoding='utf-8') as output_file:
            for line in input_file:
                bpe_line = bpe.process_line(line.strip())
                output_file.write(bpe_line + '\n')

# Example usage
if __name__ == '__main__':
    tokenizer = ParallelCorpusTokenizer()
    
    # Step 1: Tokenize parallel corpus
    tokenizer.tokenize_files(
        'english_sentences.txt', 
        'german_sentences.txt', 
        'tokenized_english.txt', 
        'tokenized_german.txt'
    )
    
    # Step 2: Learn BPE codes from tokenized files
    tokenizer.learn_bpe(
        'tokenized_english.txt', 
        'en_bpe_codes.txt'
    )
    
    # Step 3: Apply BPE to tokenized files
    tokenizer.apply_bpe(
        'tokenized_english.txt', 
        'bpe_english.txt', 
        'en_bpe_codes.txt'
    )