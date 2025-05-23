(define (domain logistics-durative)
  (:requirements :strips :typing :negative-preconditions :durative-actions :action-costs)

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

  (:durative-action load
    :parameters (?p - package ?v - vehicle ?loc - location)
    :duration (= ?duration 1)
    :condition (and
      (at start (at ?v ?loc))
      (at start (at-pkg ?p ?loc))
      (over all (at ?v ?loc))
    )
    :effect (and
      (at end (in ?p ?v))
      (at end (not (at-pkg ?p ?loc)))
    )
  )

  (:durative-action unload
    :parameters (?p - package ?v - vehicle ?loc - location)
    :duration (= ?duration 1)
    :condition (and
      (at start (at ?v ?loc))
      (at start (in ?p ?v))
      (over all (at ?v ?loc))
    )
    :effect (and
      (at end (at-pkg ?p ?loc))
      (at end (not (in ?p ?v)))
    )
  )

  (:durative-action drive
    :parameters (?v - truck ?from ?to - location)
    :duration (= ?duration 2)
    :condition (and
      (at start (at ?v ?from))
      (at start (road-connection ?from ?to))
    )
    :effect (and
      (at end (not (at ?v ?from)))
      (at end (at ?v ?to))
      (at end (increase (total-cost) (road-cost ?from ?to)))
    )
  )

  (:durative-action fly
    :parameters (?v - plane ?from ?to - location)
    :duration (= ?duration 3)
    :condition (and
      (at start (at ?v ?from))
      (at start (air-connection ?from ?to))
    )
    :effect (and
      (at end (not (at ?v ?from)))
      (at end (at ?v ?to))
      (at end (increase (total-cost) (air-cost ?from ?to)))
    )
  )

  (:durative-action sail
    :parameters (?v - ship ?from ?to - location)
    :duration (= ?duration 4)
    :condition (and
      (at start (at ?v ?from))
      (at start (water-connection ?from ?to))
    )
    :effect (and
      (at end (not (at ?v ?from)))
      (at end (at ?v ?to))
      (at end (increase (total-cost) (water-cost ?from ?to)))
    )
  )
)
