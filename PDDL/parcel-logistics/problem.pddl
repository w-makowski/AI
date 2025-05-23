(define (problem logistics-problem)
    (:domain logistics)
    
    (:objects
        city1 city2 city3 city4 hub1 port1 port2 - location
        truck1 truck2 - truck
        plane1 plane2 - plane
        ship1 - ship
        pkg1 pkg2 pkg3 - package
    )

    (:init
        (at truck1 city1)
        (at truck2 city3)
        (at plane1 city1)
        (at plane2 hub1)
        (at ship1 port1)

        (at-pkg pkg1 city1)
        (at-pkg pkg2 port1)
        (at-pkg pkg3 city3)

        (road-connection city1 city2)
        (road-connection city2 city1)
        (road-connection city2 city3)
        (road-connection city3 city2)
        (road-connection city1 city4)
        (road-connection city4 city1)
        (road-connection city4 city3)
        (road-connection city3 city4)
        (road-connection city3 city1)
        (road-connection city1 city3) 

        (air-connection city1 hub1)
        (air-connection hub1 city1)
        (air-connection hub1 city3)
        (air-connection city3 hub1)
        (air-connection city1 city3)
        (air-connection city3 city1)

        (water-connection port1 port2)
        (water-connection port2 port1)
        (water-connection city2 port2)
        (water-connection port2 city2)

        (= (road-cost city1 city2) 3)
        (= (road-cost city2 city1) 3)
        (= (road-cost city2 city3) 2)
        (= (road-cost city3 city2) 2)
        (= (road-cost city1 city4) 1)
        (= (road-cost city4 city1) 2)
        (= (road-cost city3 city4) 1)
        (= (road-cost city4 city3) 1)
        (= (road-cost city3 city1) 8)
        (= (road-cost city1 city3) 8)

        (= (air-cost city1 hub1) 4)
        (= (air-cost hub1 city1) 5)
        (= (air-cost hub1 city3) 4)
        (= (air-cost city3 hub1) 4)
        (= (air-cost city1 city3) 10)
        (= (air-cost city3 city1) 10)

        (= (water-cost port1 port2) 6)
        (= (water-cost port2 port1) 6)
        (= (water-cost city2 port2) 2)
        (= (water-cost port2 city2) 3)

        (= (total-cost) 0)
    )

    (:goal
        (and
        (at-pkg pkg1 city3)
        (at-pkg pkg2 port2)
        (at-pkg pkg3 city1)
        )
    )

    (:metric minimize (total-cost))
)
