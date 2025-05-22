import logging

from bs4 import BeautifulSoup
import requests

#logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/117.0.0.0 Safari/537.36"
}

class WebCrowler:
    '''Utility Class'''
    def __init__(self, url):
        self.url = url
        self.body = None
        self.title = "No title found"
        self.text = ""
        self.links = set()
        self.crawling_successful = False

        try:
            response = requests.get(url, headers=headers, timeout=10)
            response.raise_for_status()
            self.body = response.content
            soup = BeautifulSoup(self.body, 'html.parser')
            self.title = soup.title.string if soup.title else "No title found"
            if soup.body:
                for irrelevent in soup.body(["script", "style", "img", "input"]):
                    irrelevent.decompose()
                self.text = soup.body.get_text(separator="\n", strip=True)
            else:
                self.text = ""

            links = [ link.get('href') for link in soup.find_all('a') ]
            self.links = {link for link in links if link}
            self.crawling_successful = True
        except requests.exceptions.ConnectionError as e:
            logger.warning(f"Could not connect to {url}. Details {e} ")
        except requests.exceptions.Timeout as e:
            logger.warning(f"Request to {url} timed out. Details {e} ")
        except requests.exceptions.HTTPError as e:
            logger.warning(f"Http error for {url} timed out. Status code {e.response.status_code}. Details {e}")
        except requests.exceptions.RequestException as e:
            logger.warning(f"Unexpected Request to {url}. Details {e} ")
        except Exception as e:
            logger.warning(f"Unexpected error during crawling {url}. Details {e}")

    def get_contents(self):
        return f"Webpage Title:\n{self.title}\nWebpage Contents:{self.text}\n\n"
