(define (domain logistics)
  (:requirements :strips :typing :negative-preconditions :action-costs)

  (:types
    location vehicle package - object
    truck plane ship - vehicle
  )

  (:predicates
    (at ?v - vehicle ?l - location)
    (at-pkg ?p - package ?l - location)
    (in ?p - package ?v - vehicle)
    (road-connection ?from ?to - location)
    (air-connection ?from ?to - location)
    (water-connection ?from ?to - location)
  )

  (:functions
    (total-cost)
    (road-cost ?from ?to - location)
    (air-cost ?from ?to - location)
    (water-cost ?from ?to - location)
  )

  (:action load
    :parameters (?p - package ?v - vehicle ?loc - location)
    :precondition (and
      (at ?v ?loc)
      (at-pkg ?p ?loc)
    )
    :effect (and
      (in ?p ?v)
      (not (at-pkg ?p ?loc))
    )
  )

  (:action unload
    :parameters (?p - package ?v - vehicle ?loc - location)
    :precondition (and
      (at ?v ?loc)
      (in ?p ?v)
    )
    :effect (and
      (at-pkg ?p ?loc)
      (not (in ?p ?v))
    )
  )

  (:action drive
    :parameters (?v - truck ?from ?to - location)
    :precondition (and
      (at ?v ?from)
      (road-connection ?from ?to)
    )
    :effect (and
      (not (at ?v ?from))
      (at ?v ?to)
      (increase (total-cost) (road-cost ?from ?to))
    )
  )

  (:action fly
    :parameters (?v - plane ?from ?to - location)
    :precondition (and
      (at ?v ?from)
      (air-connection ?from ?to)
    )
    :effect (and
      (not (at ?v ?from))
      (at ?v ?to)
      (increase (total-cost) (air-cost ?from ?to))
    )
  )

  (:action sail
    :parameters (?v - ship ?from ?to - location)
    :precondition (and
      (at ?v ?from)
      (water-connection ?from ?to)
    )
    :effect (and
      (not (at ?v ?from))
      (at ?v ?to)
      (increase (total-cost) (water-cost ?from ?to))
    )
  )
)
