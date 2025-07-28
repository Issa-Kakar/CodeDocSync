from codedocsync.suggestions.rag_corpus import RAGCorpusManager


def test_semantic_name_similarity():
    manager = RAGCorpusManager()

    # Test equivalent verbs
    assert manager._calculate_semantic_name_similarity("get_user", "fetch_user") > 0.8
    assert manager._calculate_semantic_name_similarity("create_item", "make_item") > 0.8


def test_type_compatibility():
    manager = RAGCorpusManager()

    # Test exact match
    assert manager._types_compatible("str", "str")

    # Test equivalences
    assert manager._types_compatible("str", "string")
    assert manager._types_compatible("int", "integer")
