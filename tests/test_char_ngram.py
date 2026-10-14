"""CharNGramEncoder 测试。"""
from longersession.encoder.char_ngram import CharNGramEncoder


def _cos(a, b):
    return sum(x * y for x, y in zip(a, b))


def test_zh_similarity():
    e = CharNGramEncoder(n=3, dim=128)
    emb = e.encode(
        ["Docker 镜像怎么管理版本", "Docker 镜像版本控制", "数据备份频率怎么定"]
    )
    assert _cos(emb[0], emb[1]) > _cos(emb[0], emb[2])


def test_en_similarity():
    e = CharNGramEncoder(n=3, dim=128)
    emb = e.encode(
        ["Docker image management", "Docker image versioning", "data backup frequency"]
    )
    assert _cos(emb[0], emb[1]) > _cos(emb[0], emb[2])


def test_short_text():
    e = CharNGramEncoder(n=3, dim=128)
    emb = e.encode(["ok", "hi", "go"])
    assert len(emb) == 3 and len(emb[0]) == 128
