INCLUDES = -I$(SDK_DIR)/userspace/include
CFLAGS = -DCONFIG_LOGLEVEL=4 -std=gnu99 -g -w $(INCLUDES)
LDLIBS = -lfpga_mgmt -lrt -lpthread
SRC = common_dma.c test_dram_dma.c 
OBJ = $(SRC:.c=.o)

dram_dma: $(OBJ)
	gcc $(CFLAGS) -o test_dram_dma $^ $(LDFLAGS) $(LDLIBS)

experiment: experiment.py
	python experiment.py

remote_experiment: remote_experiment.py
	python remote_experiment.py

install_sshpass:
	sudo yum -y install sshpass

clean:
	rm -f *.o test_dram_dma *.pyc
	rm -rf *.csv specification*.json


