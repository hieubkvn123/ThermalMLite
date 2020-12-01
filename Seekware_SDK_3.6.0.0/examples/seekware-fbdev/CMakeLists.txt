#Version-------------------------------------------------------------------------------------------------------------------#

#Sources-------------------------------------------------------------------------------------------------------------------#

set(SOURCES 
	src/seekware-fbdev.c
)

#Project--------------------------------------------------------------------------------------------------------------------#

project(seekware-fbdev DESCRIPTION "Seekware SDK Library")

#Targets---------------------------------------------------------------------------------------------------------------------#

add_executable(${PROJECT_NAME} ${SOURCES})

set_target_properties(${PROJECT_NAME} PROPERTIES INSTALL_RPATH ../../lib/${TOOLCHAIN_DIR})

#Dependencies----------------------------------------------------------------------------------------------------------------#

#Extra Compile Options------------------------------------------------------------------------------------------------------#

#Include--------------------------------------------------------------------------------------------------------------------#

include_directories(
	${CMAKE_CURRENT_SOURCE_DIR}/include
)

#Link------------------------------------------------------------------------------------------------------------------------#

#seekware
target_link_libraries(${PROJECT_NAME} seekware)

#Install---------------------------------------------------------------------------------------------------------------------#

install(TARGETS ${PROJECT_NAME} DESTINATION bin)

file(GLOB SRC ${CMAKE_CURRENT_LIST_DIR}/src/*)
install(FILES ${SRC} DESTINATION src/${PROJECT_NAME}/src)

file(GLOB INC ${CMAKE_CURRENT_LIST_DIR}/include/*)
install(FILES ${INC} DESTINATION src/${PROJECT_NAME}/include)

install(FILES ${CMAKE_CURRENT_LIST_DIR}/Makefile DESTINATION src/${PROJECT_NAME})
