from ultimate_memory.ranking import filter_entity_scoped_results


def result(text, entities=None, speaker=None):
    return {
        "text": text,
        "provenance": {
            "entities": entities or [],
            "speaker": speaker,
        },
    }


def test_entity_scope_removes_other_people_when_enough_matches_exist():
    items = [
        result("Caroline lives in Boston.", ["Caroline"], "Caroline"),
        result("Caroline likes painting.", ["Caroline"], "Caroline"),
        result("Melanie likes pottery.", ["Melanie"], "Melanie"),
    ]
    scoped = filter_entity_scoped_results(items, ["Caroline"])
    assert len(scoped) == 2
    assert all("Melanie" not in item["text"] for item in scoped)


def test_entity_scope_falls_back_when_pool_is_too_small():
    items = [
        result("Caroline likes painting.", ["Caroline"], "Caroline"),
        result("Melanie likes pottery.", ["Melanie"], "Melanie"),
    ]
    scoped = filter_entity_scoped_results(items, ["Caroline"], min_matches=2)
    assert scoped == items
