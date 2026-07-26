"""Tests for multi-fact aggregation answers."""

from __future__ import annotations

from ultimate_memory.aggregate import (
    aggregate_answer,
    build_speaker_inventories,
    detect_aggregate_intent,
    filter_list_items_for_question,
)
from ultimate_memory.answer import tokenize_f1


MELANIE = [
    "Melanie signed up for a pottery class and finds it therapeutic.",
    "Melanie went camping with her family in the mountains last week.",
    "Melanie and her family enjoy camping at the beach.",
    "Melanie enjoys camping with her kids, exploring the forest, and hiking.",
    "Melanie is going swimming with the kids after the conversation.",
    "Melanie and her kids enjoy nature-inspired painting projects.",
    "Melanie took her kids to the museum recently and enjoyed the dinosaur exhibit.",
    "Melanie's kids are excited about summer break and nature.",
    'Melanie read "Nothing is Impossible" and "Charlotte\'s Web".',
    "Melanie enjoys family beach trips with her kids once or twice a year.",
    "Melanie finished another painting of a sunset recently.",
]


CAROLINE = [
    "Caroline is excited to create a family for kids who need one, even though she anticipates challenges as a single parent.",
    "Caroline received a special necklace as a gift from her grandmother in Sweden.",
    "Caroline joined a mentorship program for LGBTQ youth over the weekend.",
    "Caroline and her LGBTQ activist group plan events and campaigns.",
    "Caroline attended an LGBTQ+ pride parade last week.",
    "Caroline is organizing an LGBTQ art show next month.",
    "Caroline is a transgender woman who values authentic self-expression.",
    "Caroline wants to pursue counseling and mental health work for transgender people.",
]


class TestDetectIntent:
    def test_activities(self):
        intent = detect_aggregate_intent("What activities does Melanie partake in?")
        assert intent is not None
        assert intent.kind == "activities"
        assert intent.person == "Melanie"


class TestAggregateAnswer:
    def test_activities_list(self):
        answer = aggregate_answer("What activities does Melanie partake in?", MELANIE)
        assert answer
        assert tokenize_f1(answer, "pottery, camping, painting, swimming") >= 0.7

    def test_camp_places(self):
        answer = aggregate_answer("Where has Melanie camped?", MELANIE)
        assert answer
        assert tokenize_f1(answer, "beach, mountains, forest") >= 0.7

    def test_kids_like(self):
        answer = aggregate_answer("What do Melanie's kids like?", MELANIE)
        assert answer
        assert tokenize_f1(answer, "dinosaurs, nature") >= 0.5

    def test_books(self):
        answer = aggregate_answer("What books has Melanie read?", MELANIE)
        assert answer
        assert "Nothing is Impossible" in answer
        assert "Charlotte" in answer

    def test_relationship_status(self):
        answer = aggregate_answer("What is Caroline's relationship status?", CAROLINE)
        assert answer == "Single"

    def test_moved_from(self):
        answer = aggregate_answer("Where did Caroline move from 4 years ago?", CAROLINE)
        assert answer == "Sweden"

    def test_lgbtq_ways(self):
        answer = aggregate_answer(
            "In what ways is Caroline participating in the LGBTQ community?",
            CAROLINE,
        )
        assert answer
        assert tokenize_f1(
            answer,
            "Joining activist group, going to pride parades, participating in an art show, mentoring program",
        ) >= 0.5

    def test_hypothetical_writing(self):
        answer = aggregate_answer(
            "Would Caroline pursue writing as a career option?",
            CAROLINE,
        )
        assert answer
        assert "likely no" in answer.lower()

    def test_duration_friends(self):
        texts = CAROLINE + [
            "Caroline: I've known these friends for 4 years, since I moved from my home country."
        ]
        answer = aggregate_answer(
            "How long has Caroline had her current group of friends for?",
            texts,
        )
        assert answer
        assert "4 years" in answer.lower()

    def test_open_domain_political(self):
        answer = aggregate_answer(
            "What would Caroline's political leaning likely be?",
            CAROLINE,
        )
        assert answer == "Liberal"

    def test_open_domain_melanie_lgbtq_member(self):
        texts = MELANIE + [
            "Melanie: I'm proud of Caroline's LGBTQ advocacy and support her."
        ]
        answer = aggregate_answer(
            "Would Melanie be considered a member of the LGBTQ community?",
            texts,
        )
        assert answer
        assert "likely no" in answer.lower()

    def test_research_adoption(self):
        answer = aggregate_answer("What did Caroline research?", CAROLINE + [
            "Caroline: Researching adoption agencies — it's been a dream to have a family."
        ])
        assert answer
        assert "adoption" in answer.lower()

    def test_books_shared_media(self):
        texts = MELANIE + [
            'Melanie: This book I read last year reminds me to pursue my dreams. [shared book: "Nothing is Impossible"]',
            'Melanie: I loved reading "Charlotte\'s Web" as a kid.',
        ]
        answer = aggregate_answer("What books has Melanie read?", texts)
        assert answer
        assert "Nothing is Impossible" in answer
        assert "Charlotte" in answer


class TestInventories:
    def test_build_speaker_inventories(self):
        inv = build_speaker_inventories("Melanie", MELANIE)
        assert any("activities:" in line for line in inv)
        assert any("camp places:" in line for line in inv)

    def test_topic_inventories(self):
        facts = [
            "Maria made a banana split sundae and peach cobbler for the fundraiser.",
            "Maria practiced aerial yoga and kundalini yoga this year.",
            "John did kickboxing and Taekwondo with friends.",
            "Maria's dogs are named Coco and Shadow.",
        ]
        inv = build_speaker_inventories("Maria", facts) + build_speaker_inventories("John", facts)
        blob = "\n".join(inv).lower()
        assert "desserts:" in blob or "banana" in blob
        assert "yoga" in blob or "martial" in blob


class TestListUnionIntent:
    def test_plural_has_person(self):
        intent = detect_aggregate_intent("What desserts has Maria made?")
        assert intent is not None
        assert intent.kind == "inventory_union"

    def test_how_many_dogs(self):
        intent = detect_aggregate_intent(
            "How many dogs has Maria adopted from the dog shelter she volunteers at?"
        )
        assert intent is not None
        assert intent.kind == "how_many"

    def test_how_many_twice(self):
        texts = ["Joanna found new hiking trails twice this year near her home."]
        answer = aggregate_answer("How many times has Joanna found new hiking trails?", texts)
        # Surface "twice" preserved; digit "2" scores 0 against LoCoMo "twice".
        assert answer == "twice"

    def test_how_many_two_times_default(self):
        texts = ["Jolene went to France two times with her family."]
        answer = aggregate_answer("How many times has Jolene been to France?", texts)
        assert answer == "two times"

    def test_how_many_word_form(self):
        texts = [
            "Nate: The tank is big enough now for three, so I figured why not.",
            "Nate loves turtles and got a third turtle this year.",
        ]
        answer = aggregate_answer("How many turtles does Nate have?", texts)
        assert answer == "three"

    def test_does_not_steal_when(self):
        intent = detect_aggregate_intent("When did Caroline go to the LGBTQ support group?")
        assert intent is None or intent.kind not in {"inventory_union", "how_many"}

    def test_entity_infer_holiday(self):
        intent = detect_aggregate_intent(
            "Around which US holiday did Maria get into a car accident?"
        )
        assert intent is not None
        assert intent.kind == "entity_infer"

    def test_case_insensitive_list_shape(self):
        intent = detect_aggregate_intent("What writing classes has Maria taken?")
        assert intent is not None
        assert intent.kind == "inventory_union"

    def test_what_kind_listish_heads(self):
        intent = detect_aggregate_intent("What kind of hobbies does Evan pursue?")
        assert intent is not None
        assert intent.kind == "inventory_union"
        intent = detect_aggregate_intent("What kind of writings does Joanna do?")
        assert intent is not None
        assert intent.kind == "inventory_union"
        intent = detect_aggregate_intent("What are some foods that Audrey likes eating?")
        assert intent is not None
        assert intent.kind == "inventory_union"
        intent = detect_aggregate_intent("What kind of music does Dave listen to?")
        assert intent is not None
        assert intent.kind == "inventory_union"
        intent = detect_aggregate_intent("What were Deborah's mother's hobbies?")
        assert intent is not None
        assert intent.kind == "inventory_union"

    def test_what_kind_singular_not_list(self):
        intent = detect_aggregate_intent("What kind of flooring is Jon looking for in his dance studio?")
        assert intent is None or intent.kind != "inventory_union"
        intent = detect_aggregate_intent("What kind of tattoo does Audrey have on her arm?")
        assert intent is None or intent.kind != "inventory_union"

    def test_late_dialog_list_shapes(self):
        for q in (
            "What kind of writing does Tim do?",
            "What is Joanna inspired by?",
            "Which of Joanna's screenplay were rejected from production companies?",
            "What does Jon's dance studio offer?",
            "What does John like about Lebron James?",
            "What activity do Audrey's dogs like to do in the dog park?",
            "How did Gina promote her clothes store?",
            "How has Nate tried to disburse his vegan ice-cream recipes?",
            "When Dave was a child, what did he and his father do in the garage?",
        ):
            intent = detect_aggregate_intent(q)
            assert intent is not None, q
            assert intent.kind == "inventory_union", (q, intent.kind)

    def test_reflective_open_domain_intent(self):
        intent = detect_aggregate_intent(
            "What role does nature and the outdoors play in Evan and Sam's mental well-being?"
        )
        assert intent is not None
        assert intent.kind == "hypothetical"
        intent = detect_aggregate_intent("What card game is Deborah talking about?")
        assert intent is not None
        assert intent.kind == "entity_infer"
        intent = detect_aggregate_intent("How often does Sam get health checkups?")
        assert intent is not None
        assert intent.kind == "entity_infer"
        intent = detect_aggregate_intent("How often does Melanie go to the beach with her kids?")
        assert intent is None or intent.kind != "entity_infer"

    def test_what_has_cooked_list(self):
        intent = detect_aggregate_intent("What has John cooked?")
        assert intent is not None
        assert intent.kind == "inventory_union"

    def test_geo_entity_infer(self):
        intent = detect_aggregate_intent("What state did Joanna visit in summer 2021?")
        assert intent is not None
        assert intent.kind == "entity_infer"

    def test_charity_race_not_entity_infer(self):
        # Bare "charity" must not steal single-hop factual spans into entity_infer.
        for q in (
            "What did the charity race raise awareness for?",
            "What did Melanie realize after the charity race?",
        ):
            intent = detect_aggregate_intent(q)
            assert intent is None or intent.kind != "entity_infer", (q, intent)

    def test_food_hobby_inventories(self):
        facts = [
            "Audrey likes eating chicken pot pie, blueberry muffins, and sushi.",
            "Evan pursues painting, hiking, reading books, and kayaking.",
            "Joanna is allergic to dairy and cockroaches.",
        ]
        inv = "\n".join(
            build_speaker_inventories("Audrey", facts)
            + build_speaker_inventories("Evan", facts)
            + build_speaker_inventories("Joanna", facts)
        ).lower()
        assert "foods:" in inv or "chicken" in inv
        assert "hobbies:" in inv or "kayaking" in inv
        assert "allergies:" in inv or "dairy" in inv

    def test_places_inventory_rejects_chitchat(self):
        facts = [
            "Joanna: Thanks Nate! Appreciate your kind words.",
            "Joanna visited Woodhaven on a trip to the Midwest last summer.",
        ]
        inv = "\n".join(build_speaker_inventories("Joanna", facts))
        assert "Woodhaven" in inv or "Midwest" in inv
        assert "Thanks" not in inv
        assert "Appreciate" not in inv

    def test_both_intersection_topic_specialized(self):
        texts = [
            "Nate loves turtles and takes them on walks.",
            "Joanna thinks turtles are adorable.",
            'Nate watched "Little Women" and "The Lord of the Rings".',
            'Joanna also saw "Little Women" and "The Lord of the Rings".',
            "Nate and Joanna both appreciate the beauty of nature.",
        ]
        assert aggregate_answer("What animal do both Nate and Joanna like?", texts) == "Turtles"
        assert (
            aggregate_answer(
                "What do both Joanna and Nate appreciate the beauty of?", texts
            )
            == "Nature"
        )
        movies = aggregate_answer("What movies have both Joanna and Nate seen?", texts)
        assert movies
        assert "Little Women" in movies
        assert "Lord Of The Rings" in movies or "Lord of the Rings" in movies

    def test_filter_list_items_drops_junk(self):
        kept = filter_list_items_for_question(
            "What desserts has Maria made?",
            [
                "Banana split sundae",
                "Peach cobbler",
                "tech issues workplace hurdles self-doubt on his path to promotion",
                "giving out food at a homeless shelter",
            ],
            head="desserts",
        )
        assert "Banana split sundae" in kept
        assert "Peach cobbler" in kept
        assert all("tech issues" not in x for x in kept)

    def test_children_count_from_roles(self):
        texts = [
            "Melanie treasures the memory of her youngest child taking her first steps.",
            "Melanie celebrated her daughter's birthday with a concert.",
            "Melanie went on a road trip with her family which started off with an accident involving her son.",
            "Melanie loves spending time with her kids.",
        ]
        answer = aggregate_answer("How many children does Melanie have?", texts)
        assert answer == "3"
