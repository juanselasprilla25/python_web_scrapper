import argparse
import logging

logging.basicConfig(level=logging.INFO)

import pandas as pd
import hashlib

from urllib.parse import urlparse

import nltk
from nltk.corpus import stopwords


logger = logging.getLogger(__name__)

def main(filename):
	logger.info('Starting cleaning process')

	df = _read_data_(filename)
	newspaper_uid = _extract_newspaper_uid(filename)
	df = _add_newspaper_uid_column(df, newspaper_uid)
	df = _extract_host(df)
	df = _fill_missing_titles(df)
	df = _generate_uids_rows(df)
	df = _remove_line_jumps(df)
	df = _tokenize_column(df, 'body')
	df = _tokenize_column(df, 'title')
	df = _remove_duplicates(df, 'title')
	df = _drops_rows_missing_data(df)
	_save_data(df, filename)

	return df

def _read_data_(filename):
	logger.info('Reading File {}'.format(filename))

	return pd.read_csv(filename)

def _extract_newspaper_uid(filename):
	logger.info('Extracting Newspaper UID')
	newspaper_uid = filename.split('_')[1]

	logger.info('Newspaper UID Detected: {}'.format(newspaper_uid))
	return	newspaper_uid

def _add_newspaper_uid_column(df, newspaper_uid):
	logger.info('Filling Newspaper_UID Column With {}'.format(newspaper_uid))
	df['newspaper_uid'] = newspaper_uid

	return df

def _extract_host(df):
	logger.info('Extracting HOST From URL')
	df['host'] = df['url'].apply(lambda url: urlparse(url).netloc)

	return df

def _fill_missing_titles(df):
	logger.info('Filling Missing Titles')
	missing_title_mask = df['title'].isna()

	missing_titles = (df[missing_title_mask]['url']
                		.str.extract(r'(?P<missing_titles>[^/]+)$')
                    	.applymap(lambda title: title.split('-'))
                    	.applymap(lambda title_word_list: ' '.join(title_word_list))
                 	 )

	df.loc[missing_title_mask, 'title'] = missing_titles.loc[:, 'missing_titles']

	return df

def _generate_uids_rows(df):
	logger.info('Generating UID For Rows')
	uids = (df
            .apply(lambda row: hashlib.md5(bytes(row['url'].encode())), axis = 1)
            .apply(lambda hash_object: hash_object.hexdigest())
       )

	df['uid'] = uids

	return df.set_index('uid')

def _remove_line_jumps(df):
	logger.info('Deleting Line Jumps From Data')
	stripped_body = ( df
                     .apply(lambda row: row['body'], axis = 1)
                     .apply(lambda body: list(body))
                     .apply(lambda letters: list(map(lambda letter: letter.replace('\n', ''), letters)))
                     .apply(lambda letters: ''.join(letters))
               		)

	df['body'] = stripped_body

	return df

def _tokenize_column(df, column_name):
	logger.info('Counting Significative Words in Column: {}'.format(column_name))
	stop_words = set(stopwords.words('spanish'))

	words_count =	( df
                		.dropna()
                		.apply(lambda row: nltk.word_tokenize(row[column_name]), axis = 1)
                		.apply(lambda tokens: list(filter(lambda token: token.isalpha(), tokens)))
                		.apply(lambda tokens: list(map(lambda token: token.lower(), tokens)))
                		# .apply(lambda count: len(count))
                		# Borrando las siguientes dos líneas de código y usando COUNT se encuentra
                		# el número total de tokens encontrados
                		.apply(lambda word_list: list(filter(lambda word: word not in stop_words, word_list)))

                		.apply(lambda valid_word_list: len(valid_word_list))
           			)

	df['n_tokens_' + column_name ] = words_count

	return df

def _remove_duplicates(df, column_name):
	logger.info('Deleting Duplicate Registers')

	df.drop_duplicates(subset = [column_name], keep = 'first', inplace = True)

	return df

def _drops_rows_missing_data(df):
	logger.info('Deleting Rows With Missing Data')

	return df.dropna()

def _save_data(df, filename):
	# Crear nueva carpeta "Clean_Data_Sets" y guardar archivos "clean_..."
	clean_filename = 'clean_{}'.format(filename)
	logger.info('Saving Data At: {}'.format(clean_filename))
	df.to_csv(clean_filename)

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('filename',
						help = 'PATH to the dirty data',
						type = str)
	args = parser.parse_args()
	df = main(args.filename)

	print(df)
