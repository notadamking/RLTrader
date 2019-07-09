# -*- mode: ruby -*-
# vi: set ft=ruby :

Vagrant.require_version '>= 2.2'

VAGRANTFILE_API_VERSION = '2'

Vagrant.configure(VAGRANTFILE_API_VERSION) do |config|
    config.ssh.forward_x11 = true
    machine_ip_address = '192.168.181.21' #random IP for low chance of collision

    required_plugins = %w(vagrant-vbguest vagrant-hostmanager)

    # Install plugins if missing
    plugins_to_install = required_plugins.select {|plugin| not Vagrant.has_plugin? plugin}
    if plugins_to_install.any?
        puts "Installing plugins: #{plugins_to_install.join(' ')}"
        if system "vagrant plugin install #{plugins_to_install.join(' ')}"
            exec "vagrant #{ARGV.join(' ')}"
        else
        abort 'Installation of one or more plugins has failed. Aborting.'
        end
    end

    # Configure hosts
    if Vagrant.has_plugin?('vagrant-hostmanager')
        config.hostmanager.enabled = true
        config.hostmanager.manage_host = true
        config.hostmanager.manage_guest = true
        config.hostmanager.ignore_private_ip = false
        config.hostmanager.aliases = ['trader-rl.local']
    end

    # Set auto_update to false, if you do NOT want to check the correct virtual-box-guest-additions version when booting VM
    if Vagrant.has_plugin?('vagrant-vbguest')
        config.vbguest.auto_update = false
    end

    config.vm.define 'trader-rl-vagrant', primary: true do |vm_config|
        vm_config.vm.box = 'ubuntu/bionic64'
        vm_config.vm.box_check_update = true
        vm_config.vm.network 'private_network', ip: machine_ip_address
        vm_config.vm.provider 'virtualbox' do |vb|
        vb.name = 'trader-rl'
        vb.cpus = 8
        vb.memory = 20480
    end

    vm_config.vm.hostname = 'trader-rl'
    vm_config.ssh.insert_key = false

    vm_config.vm.synced_folder '.', '/vagrant', disabled: false
vm_config.vm.provision "default setup", type: "shell", inline: <<SCRIPT
apt update
apt install mpich libpq-dev
DEBIAN_FRONTEND=noninteractive apt install python3-pip
pip3 install -r /vagrant/requirements.no-gpu.txt
SCRIPT

vm_config.vm.provision "Convenience method", type: "shell", privileged: false, inline: <<SCRIPT
grep -q 'cd /vagrant' ~/.bashrc || echo -e "\n\ncd /vagrant" > ~/.bashrc
SCRIPT

    end
end
