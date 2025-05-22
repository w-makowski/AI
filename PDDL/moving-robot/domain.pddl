(define (domain moving-robot)
    (:requirements :strips :typing)
    (:types robot room ball arm)

    (:predicates
        (at ?r - robot ?rm - room)
        (inroom ?b - ball ?rm - room)
        (holding ?a - arm ?b - ball)
        (arm-empty ?a - arm)
        (connected ?rm1 ?rm2 - room)
    )

    (:action move
        :parameters (?r - robot ?from ?to - room)
        :precondition (and
            (at ?r ?from)
            (connected ?from ?to)
        )
        :effect (and
            (not (at ?r ?from))
            (at ?r ?to)
        )
    )

    (:action pick-up
        :parameters (?r - robot ?a - arm ?b - ball ?rm - room)
        :precondition (and
            (at ?r ?rm)
            (inroom ?b ?rm)
            (arm-empty ?a)
        )
        :effect (and
            (holding ?a ?b)
            (not (arm-empty ?a))
            (not (inroom ?b ?rm))
        )
    )

    (:action put-down
        :parameters (?r - robot ?a - arm ?b - ball ?rm - room)
        :precondition (and
            (at ?r ?rm)
            (holding ?a ?b)
        )
        :effect (and
            (inroom ?b ?rm)
            (arm-empty ?a)
            (not (holding ?a ?b))
        )
    )
)
