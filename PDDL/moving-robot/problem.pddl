(define (problem move-balls)
    (:domain moving-robot)

    (:objects
        room1 room2 room3 - room
        robot - robot
        ball1 ball2 ball3 ball4 - ball
        arm1 arm2 - arm
    )

    (:init
        (at robot room1)
        (inroom ball1 room1)
        (inroom ball2 room1)
        (inroom ball3 room2)
        (inroom ball4 room1)
        (arm-empty arm1)
        (arm-empty arm2)
        (connected room1 room2)
        (connected room2 room1)
        (connected room2 room3)
        (connected room3 room2)
    )

    (:goal
        (and
            (inroom ball1 room3)
            (inroom ball2 room3)
            (inroom ball3 room3)
            (inroom ball4 room3)
        )
    )
)
