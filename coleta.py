import configparser
import json
import praw
import re
from datetime import datetime

##Ler as credenciais

config = configparser.ConfigParser()
config.read('config.ini')

## Autenticação com a API do Reddit usando PRAW

reddit = praw.Reddit(client_id=config['REDDIT']['client_id'],
                     client_secret=config['REDDIT']['client_secret'],
                     user_agent=config['REDDIT']['user_agent'],
                     username=config['REDDIT']['username'],
                     password=config['REDDIT']['password'])


# Função para pesquisar posts em um subreddit
def search_posts(subreddit_name, post_limit, search_queries, output_file):
    
    posts = [] 
    
    for search_query in search_queries:
        # Obter o objeto do subreddit
        subreddit = reddit.subreddit(subreddit_name)
    
    # Realizar a pesquisa de posts no subreddit para cada consulta
        search_results = subreddit.search(search_query, limit=post_limit)

        # Extrair informações relevantes dos resultados
        for result in search_results:
            post = {
                'title': result.title,
                'selftext': result.selftext,
                'created_date': datetime(result.created_utc).strftime('%Y-%m-%d %H:%M:%S'),
                'subreddit': result.subreddit.display_name,
                'search_query': search_query, 
            }
            posts.append(post)
        
        print(f"Consulta '{search_query}' concluída com {len(posts)} posts coletados até agora.")

    # Salvar todos os posts em um único arquivo JSON
    with open(output_file, 'w') as f:
        json.dump(posts, f, indent=4)




# Testar a função com uma entrada de exemplo
if __name__ == "__main__":

    subreddit_name ='language'
    post_limit =1000
    search_queries =['Python','Java','C#'] 
    output_file = 'posts.json'

    search_posts(subreddit_name, post_limit, search_queries, output_file)