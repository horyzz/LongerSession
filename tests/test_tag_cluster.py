"""TagClusterer 增量聚类测试。"""
from longersession.encoder.mock import MockEncoder
from longersession.index.tag_cluster import TagClusterer


def test_new_topic_creates_tag():
    c = TagClusterer(MockEncoder(dim=16), threshold=0.6)
    assert c.add("Docker 镜像管理") == "Docker 镜像管理"


def test_similar_topic_joins_existing_tag():
    c = TagClusterer(MockEncoder(dim=16), threshold=0.6)
    c.add("Docker 镜像管理")
    assert c.add("Docker 镜像版本") == "Docker 镜像管理"


def test_dissimilar_topic_creates_new_tag():
    c = TagClusterer(MockEncoder(dim=16), threshold=0.6)
    c.add("Docker 镜像管理")
    assert c.add("数据备份策略") == "数据备份策略"


def test_anchor_is_stable():
    c = TagClusterer(MockEncoder(dim=16), threshold=0.6)
    c.add("Docker 镜像管理")
    c.add("Docker 镜像版本")
    c.add("Docker 部署流程")
    # 锚点始终是第一个进入的 topic，不随成员增加改变
    assert c.tag_of("Docker 部署流程") == "Docker 镜像管理"
    assert "Docker 镜像管理" in c.all_tags()
