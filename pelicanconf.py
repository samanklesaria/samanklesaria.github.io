AUTHOR = 'Sam Anklesaria'
SITENAME = "Sam's Blog"
SITEURL = ""

PATH = "content"

TIMEZONE = 'US/Central'

DEFAULT_LANG = 'en'

# Feed generation is usually not desired when developing
FEED_ALL_ATOM = None
CATEGORY_FEED_ATOM = None
TRANSLATION_FEED_ATOM = None
AUTHOR_FEED_ATOM = None
AUTHOR_FEED_RSS = None

MARKDOWN = {
    'extensions': [
        'fenced_code',
        'codehilite',
        'tables'
    ],
    'extension_configs': {
        'codehilite': {'guess_lang': False}
    }
}

# Blogroll
LINKS = ()

# Social widget
SOCIAL = ()

# Uncomment following line if you want document-relative URLs when developing
# RELATIVE_URLS = True
