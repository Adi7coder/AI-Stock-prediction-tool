import scrapy
from datetime import datetime

class NewsSpider(scrapy.Spider):
    name = "news"
    allowed_domains = ["economictimes.indiatimes.com"]
    start_urls = [
        "https://economictimes.indiatimes.com/markets/stocks/news"
    ]

    def parse(self, response):
        today = datetime.today().strftime("%Y-%m-%d")
        for article in response.css("div.eachStory"):
            yield {
                'publishedAt': today,  # use today as fallback
                'title': article.css('h3 a::text').get(),
                'link': response.urljoin(article.css('h3 a::attr(href)').get()),
            }
        
        next_page = response.css("a[rel=next]::attr(href)").get()
        if next_page:
            yield response.follow(next_page, self.parse)
