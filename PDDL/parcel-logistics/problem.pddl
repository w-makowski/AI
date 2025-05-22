(define (problem logistics-problem)
    (:domain logistics)
    (:objects
        magazyn-wwa centrala-wwa lotnisko-wwa - location
        magazyn-krk centrala-krk lotnisko-krk - location
    
        warszawa krakow - city
    
        ciezarowka1 ciezarowka2 samolot - vehicle
    
        paczka1 paczka2 paczka3 - package
    )
  
    (:init

        (in-city magazyn-wwa warszawa)
        (in-city centrala-wwa warszawa)
        (in-city lotnisko-wwa warszawa)
        
        (in-city magazyn-krk krakow)
        (in-city centrala-krk krakow)
        (in-city lotnisko-krk krakow)
        
        (connected magazyn-wwa centrala-wwa)
        (connected centrala-wwa magazyn-wwa)
        (connected centrala-wwa lotnisko-wwa)
        (connected lotnisko-wwa centrala-wwa)
        
        (connected magazyn-krk centrala-krk)
        (connected centrala-krk magazyn-krk)
        (connected centrala-krk lotnisko-krk)
        (connected lotnisko-krk centrala-krk)
        
        (connected lotnisko-wwa lotnisko-krk)
        (connected lotnisko-krk lotnisko-wwa)
    
        (at-vehicle ciezarowka1 magazyn-wwa)
        (at-vehicle ciezarowka2 magazyn-krk)
        (at-vehicle samolot lotnisko-wwa)
    
        (at paczka1 magazyn-wwa)
        (at paczka2 magazyn-wwa)
        (at paczka3 centrala-wwa)
    )
  
    (:goal
        (and
            (at paczka1 centrala-krk)
            (at paczka2 magazyn-krk)
            (at paczka3 lotnisko-krk)
        )
    )
)
