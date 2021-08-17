import RAKE

def get_rake():
  rake = RAKE.Rake("./assets/pt-stopwords.txt")
  return rake;