(define (domain cleaning-robot)
    (:requirements :strips :typing :negative-preconditions)
    (:types
        robot
        room
    )

    (:predicates
        (at ?r - robot ?rm - room)
        (dirty ?rm - room)
        (clean ?rm - room)
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

    (:action clean
        :parameters (?r - robot ?rm - room)
        :precondition (and
            (at ?r ?rm)
            (dirty ?rm)
            (not (clean ?rm))
        )
        :effect (and
            (not (dirty ?rm))
            (clean ?rm)
        )
    )
)
