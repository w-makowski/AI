(define (domain logistics)
    (:requirements :strips :typing :negative-preconditions :action-costs :durative-actions :conditional-effects)
    (:types
        package vehicle location truck plane   
    )

    (:predicates
        (at ?obj - (either package vehicle) ?loc - location)
        (in ?p - package ?v - vehicle)
        (road-connection ?from ?to - location)
        (air-connection ?from ?to - location)
        (available ?v - vehicle)
    )

    (:functions
        (total-cost)
    )

    (:action load
        :parameters (?p - package ?v - vehicle ?l - location)
        :precondition (and
            (at ?p ?l)
            (at-vehicle ?v ?l)
        )
        :effect (and
            (not (at ?p ?l))
            (in ?p ?v)
        )
    )

    (:action unload
        :parameters (?p - package ?v - vehicle ?l - location)
        :precondition (and
            (in ?p ?v)
            (at-vehicle ?v ?l)
        )
        :effect (and
            (not (in ?p ?v))
            (at ?p ?l)
        )
    )

    (:action drive
        :parameters (?v - vehicle ?l1 ?l2 - location)
        :precondition (and
            (at-vehicle ?v ?l1)
            (connected ?l1 ?l2)
        )
        :effect (and
            (not (at-vehicle ?v ?l1))
            (at-vehicle ?v ?l2)
        )
    )

    (:action drive-within-city
        :parameters (?v - vehicle ?l1 ?l2 - location ?c - city)
        :precondition (and
            (at-vehicle ?v ?l1)
            (in-city ?l1 ?c)
            (in-city ?l2 ?c)
        )
        :effect (and
            (not (at-vehicle ?v ?l1))
            (at-vehicle ?v ?l2)
        )
    )

)