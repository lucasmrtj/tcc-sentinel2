#!/usr/bin/env python
# coding: utf-8

# In[14]:


import urllib.error
import urllib.request
import requests
from bs4 import BeautifulSoup
from pathlib import Path
from tqdm import tqdm
import time

# URL of data set
url = 'https://jeodpp.jrc.ec.europa.eu/ftp/jrc-opendata/DRLL/AI4BOUNDARIES/'


# In[5]:


def download_file(url, dst_path):
    try:
        with urllib.request.urlopen(url) as web_file:
            data = web_file.read()
            with open(dst_path, mode='wb') as local_file:
                local_file.write(data)
    except urllib.error.URLError as e:
        print(e)


def download_ai4boundaries(dir):

    url = 'http://jeodpp.jrc.ec.europa.eu/ftp/jrc-opendata/DRLL/AI4BOUNDARIES/'
    urls = []
    url_fns = []

    def scrape(site):

        # getting the request from url
        r = requests.get(site)

        # converting the text
        s = BeautifulSoup(r.text, "html.parser")

        for i in s.find_all("a"):
            href = i.attrs['href']

            if href.endswith("/"):

                subsite = site + href

                if subsite not in urls:
                    urls.append(subsite)

                    # calling it self
                    scrape(subsite)
            if href.endswith("tif") | href.endswith("nc"):
                url_fn_ = site + href
                url_fns.append(url_fn_)

    print('Scraping data')
    scrape(url)

    print('Creating folder architecture')
    if dir.endswith('/'):
        subdirs = [i.replace(url, dir) for i in urls if not i.endswith('DRLL/')]
    else:
        subdirs = [i.replace(url, dir + '/') for i in urls if not i.endswith('DRLL/')]

    subdirs = [subdir.replace('DRLL/', '') for subdir in subdirs if not 'ftp' in subdir]

    for subdir in subdirs:
        Path(subdir).mkdir(parents=True, exist_ok=True)

    failed_fns = []
    print('Downloading data')
    for url_fn in tqdm(url_fns):
        if dir.endswith('/'):
            fn = url_fn.replace(url, dir)
        else:
            fn = url_fn.replace(url, dir + '/')
        try:
            download_file(url_fn, fn)
        except:
            time.sleep(20)
            failed_fns = url_fn

    for url_fn in tqdm(failed_fns):
        if dir.endswith('/'):
            fn = url_fn.replace(url, dir)
        else:
            fn = url_fn.replace(url, dir + '/')
        try:
            download_file(url_fn, fn)
        except:
            continue

    print('Download finished!')
    print('Cite the data set:')
    print('d\'Andrimont, R., Claverie, M., Kempeneers, P., Muraro, D., Yordanov, M., Peressutti, D., Batič, M., '
          'and Waldner, F.: AI4Boundaries: an open AI-ready dataset to map field boundaries with Sentinel-2 and aerial '
          'photography, Earth Syst. Sci. Data Discuss. [preprint], '
          'https://doi.org/10.5194/essd-2022-298, in review, 2022.')


# In[17]:


def download_file(url, dst_path):
    # Adicionamos um cabeçalho para fingir que somos um navegador normal, evitando bloqueios
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/114.0.0.0 Safari/537.36'
    }

    # Usamos o requests com stream=True para baixar sem sobrecarregar a memória
    response = requests.get(url, stream=True, headers=headers, timeout=60)
    response.raise_for_status() # Verifica se a conexão foi aceita (se não deu erro 403, 404, etc)

    # Prevenção extra: garante absolutamente que a subpasta deste arquivo existe
    Path(dst_path).parent.mkdir(parents=True, exist_ok=True)

    # Salva o arquivo em pedaços de 1 Megabyte
    with open(dst_path, mode='wb') as local_file:
        for chunk in response.iter_content(chunk_size=1024 * 1024):
            if chunk:
                local_file.write(chunk)

def download_ai4boundaries(dir):

    url = 'https://jeodpp.jrc.ec.europa.eu/ftp/jrc-opendata/DRLL/AI4BOUNDARIES/'
    urls = []
    url_fns = []

    def scrape(site):
        try:
            r = requests.get(site)
            r.raise_for_status() # Garante que não vamos raspar uma página de erro 404
        except requests.exceptions.RequestException as e:
            print(f"Erro ao acessar {site}: {e}")
            return

        s = BeautifulSoup(r.text, "html.parser")

        for i in s.find_all("a"):
            # Alguns links podem não ter href, isso previne o código de quebrar
            if 'href' not in i.attrs:
                continue

            href = i.attrs['href']

            # Evita links de "voltar" (Parent Directory) que causam loop infinito
            if href.startswith('?C=') or href.startswith('/'):
                continue

            if href.endswith("/"):
                subsite = site + href
                if subsite not in urls:
                    urls.append(subsite)
                    scrape(subsite) # recursão

            # CORREÇÃO 3: Melhor forma de checar múltiplas extensões
            elif href.endswith((".tif", ".nc")):
                url_fn_ = site + href
                url_fns.append(url_fn_)

    print('Scraping data...')
    scrape(url)

    if not url_fns:
        print("Nenhum arquivo .tif ou .nc foi encontrado. Verifique se o site está online e acessível.")
        return

    print(f'Encontrados {len(url_fns)} arquivos. Criando arquitetura de pastas...')

    # Garantindo que o diretório termina com '/'
    if not dir.endswith('/'):
        dir = dir + '/'

    subdirs = [i.replace(url, dir) for i in urls if not i.endswith('DRLL/')]
    subdirs = [subdir.replace('DRLL/', '') for subdir in subdirs if 'ftp' not in subdir]

    for subdir in subdirs:
        Path(subdir).mkdir(parents=True, exist_ok=True)

    failed_fns = []
    print('Downloading data...')
    for url_fn in tqdm(url_fns):
        fn = url_fn.replace(url, dir)

        # SEGREDO PARA ECONOMIZAR TEMPO: Pula o arquivo se ele já existe e não está vazio
        if Path(fn).exists() and Path(fn).stat().st_size > 0:
            continue

        try:
            download_file(url_fn, fn)
        except Exception as e:
            print(f"\n[Erro: {e}]") # Agora o terminal vai nos contar exatamente o que deu errado!
            print(f"Falha em {url_fn}. Tentaremos novamente no final.")
            time.sleep(2)
            failed_fns.append(url_fn)

    print('\nDownload finished!')
    print('Cite the data set:')
    print("d'Andrimont, R., Claverie, M., Kempeneers, P., Muraro, D., Yordanov, M., Peressutti, D., Batič, M., "
          "and Waldner, F.: AI4Boundaries: an open AI-ready dataset to map field boundaries with Sentinel-2 and aerial "
          "photography, Earth Syst. Sci. Data Discuss. [preprint], "
          "https://doi.org/10.5194/essd-2022-298, in review, 2022.")


# In[ ]:


out_dir = r'/home/ubuntu/tcc-sentinel2/dataset'
download_ai4boundaries(out_dir)

