#!/bin/sh

_CAD=$(cd $(dirname $0); pwd)                   
#host_dir=`pwd`                                
proc_name="main.py"                           
conf_file=""
start_param=
full_proc=${_CAD}/${proc_name}              
#full_proc=${proc_name}              
data_param=Def_Path
port_param=Def_Port
pid=0
chmod +x ${full_proc}
log_file=${_CAD}/${proc_name}.log       
#log_file=${_CAD}/nohup.out       
m_log_file=${_CAD}/${proc_name}-monitor.log

monitor_interval=10


proc_num() {
        num=`ps -ef | grep -w ${full_proc} | grep -v grep | wc -l`
	echo $num
        return $num
}
proc_id() {
        pid=`ps -ef | grep -w ${full_proc} | grep -v grep | awk '{print $2}'`
}

check_data_exist() {
        dp=`echo ${data_param}`&&dpp=\$$dp&&dppp=`eval echo $dpp`
        #echo $dppp
        if [ ! -z $dppp -a -d $dppp ];then 
                if [ ! -z "`ls $dppp`" ];then
                        return 1
                else
                        echo "$proc_name data: ($dppp) is empty."               
                        return 0
                fi
        else
                echo "$proc_name data: ($dppp) is not exist."
                return 0
        fi

}
check_port_exist() {
        #dp=`echo ${port_param}`&&dpp=\$$dp&&dppp=`eval echo $dpp`
	dppp=`cat $conf_file|grep ^$port_param |awk -F = '{print $NF}'` 
       # echo $dppp
        num=`netstat -ant|grep -w ${dppp}|grep LISTEN|wc -l`
        if [ $num -gt 0 ]; then  
                echo "$proc_name connect_port: ($engine_port) is exist. "
                return 1
        else
                return 0
        fi
}

start() {
        #ps -ef | grep $proc_name | grep -v grep
#        check_data_exist        
#        if [ $? -eq 0 ];then
#                echo $proc_name start failed.
#                exit 1
#        fi

#        check_port_exist       
#        if [ $? -eq 1 ]; then
#                echo $proc_name start failed.
#                exit 1
#        fi

        proc_num
        if [ $? -eq 0 ]; then
                nohup python -u  ${full_proc} ${start_param} >> ${log_file} 2>&1 &
                sleep 0.1
                proc_id
                dp=`echo ${port_param}`&&dpp=\$$dp&&dppp=`eval echo $dpp`
                echo $proc_name pid=${pid} port=${dppp} start success  [`date "+%F %T"`] >> ${m_log_file}     
                echo $proc_name pid=${pid} port=${dppp} start success. [`date "+%F %T"`]
        else
                proc_id
                echo $proc_name pid=${pid} alreay started.
        fi 
}
stop(){
        proc_num
        if [ $? -eq 0 ]; then                           
                echo $proc_name not started.
        else
                proc_id
                kill -9 ${pid}
                echo $proc_name pid=${pid} has been shutdown. [`date "+%F %T"`]
        fi
}


status() {
        ps -ef|grep -w ${full_proc} |grep -v grep
        dp=`echo ${port_param}`&&dpp=\$$dp&&dppp=`eval echo $dpp` 
        netstat -ant|grep -w ${dppp}|grep LISTEN
}
info(){
        tail -fn200 ${log_file}
}
monitor(){
        while [ 1 ]; do
                proc_num
                number=$?
                if [ $number -eq 0 ]; then
                        start
                        echo "$proc_name start at `date "+%F %T"`" automatically. >> $m_log_file
                fi
                sleep $monitor_interval
        done
}

case "$1" in
        start) start;;
        stop) stop;;
        restart) stop;sleep 1;start;;
		status) status;;
        monitor) monitor;;
        info) info;;
        *) echo "usage: $0 start|stop|restart|status|info";;
esac
