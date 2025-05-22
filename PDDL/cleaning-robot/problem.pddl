(define (problem cleaning-problem)
    (:domain cleaning-robot)
    (:objects
        robo - robot
        room1 room2 room3 - room
    )

    (:init
        (at robo room1)

        (dirty room1)
        (dirty room2)
        (dirty room3)

        (connected room1 room2)
        (connected room2 room1)
        (connected room2 room3)
        (connected room3 room2)
    )

    (:goal
        (and
            (clean room1)
            (clean room2)
            (clean room3)
        )
    )
)
