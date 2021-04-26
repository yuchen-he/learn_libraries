# Do not need cmake version definition because we had it in upper folder

file(GLOB SRC_TEST_FILES
	"*.cpp"
)

#add_executable(CPP_Learning_Test
#	${SRC_TEST_FILES}
#)

add_library(CPP_Learning_Test
	${SRC_TEST_FILES}
)

