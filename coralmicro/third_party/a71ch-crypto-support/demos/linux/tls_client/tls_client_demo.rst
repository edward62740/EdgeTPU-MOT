..
    Copyright 2019,2020 NXP


.. _tls-client-example:

========================================================
 OpenSSL Engine: TLS Client example for iMX/Rpi3
========================================================

- DocRevision : 0.94
- Date        : 2020-01-14

This section explains how to set-up a TLS link using the SE050 OpenSSL Engine on the client side.
A note at the bottom of this page (:numref:`tls-client-using-a71ch`) highlights the changes in case one uses an A71CH secure element.


Summary
========================================================

The TLS demo demonstrates setting up a mutually authenticated and encrypted
link between a client and a server system. The keypair used to identify the
client is stored in the Secure Element. The keypair used to identify the
server is simply available as a pem file.

The public keys associated with the respective key pairs are contained in
respectively a client and a server certificate.

The CA is a self-signed certificate. The same CA is used to sign client and
server certificate.

One can choose the keymaterial (CA, Client and Server) to be either RSA (4096 CA - 2048 client/server)
or EC (prime256v1).

The TLS demo comes in two flavours:

1. **Flavour-A:** The standard OpenSSL tools `s_server` and `s_client` are
   used to set-up and demonstrate the TLS link. The certificates used are
   simply stored on the file system of the host. The client uses the keypair
   stored inside the SE050.

2. **Flavour-B:** A TLS client program - included in source code
   (tlsSe050Client.cpp) - retrieves the client certificate from the SE050 and
   uses the keypair stored inside the SE050. It establishes a TLS connection
   with the server process `s_server`.

Steps in :numref:`credential-prep` to :numref:`startup-the-server` are
identical for the two demo flavours.

.. _credential-prep:

Credential preparation (execute once) [Optional]
========================================================

::

    > cd demos/linux/tls_client
    > ./scripts/createTlsCredentials_Optional.sh <ECC|RSA>

.. note::

    The Host SW package comes bundled with the required credentials. The
    `createTlsCredentials_Optional.sh` script will re-create equivalent
    credentials (but with new/different keypairs)

The script creates all demo required client and server credentials on the
client platform. One must transfer the server credentials to the server
platform.

Secure Element preparation (client side)
========================================================

For the purpose of the demo one MUST inject the TLS client key pair and
certificate into the Secure Element and create a reference pem file referring
to the provisioned key pair::

    > cd demos/linux/tls_client/scripts
    > python3 provisionTlsClient.py --key_type <ecc|rsa>

Further details on using these scripts can be found in the following:

provisionTlsClient.py
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. include:: scripts/doc/provisionTlsClient.rst.txt


Server side preparation
========================================================

.. note::

    The Host SW package comes bundled with the required server credentials.

Ensure the default server credentials or those created under (:numref:`credential-prep`) are available
on the server platform.

.. _startup-the-server:

Start up the server
========================================================

.. note::

    The server can run e.g. on a PC. The server must be reacheable over the
    TCP/IP network for the Client. Choose either a server using EC based credentials or
    a server using RSA based credentials.

Execute the following command on the server platform to use the EC based server credentials, make a note on the IP
address of the server::

    > cd demos/linux/tls_client/scripts
    > ./tlsServer.sh <ECDHE|ECDHE_SHA256|max>

Execute the following command on the server platform to use the RSA based server credentials, make a note on the IP
address of the server::

    > cd demos/linux/tls_client/scripts
    > ./tlsServer.sh RSA

Establish a TLS link from the client to the server
========================================================

The client process establishing the TLS connection comes in two flavours:
either `s_client` or a program provided in source code (tlsSe050Client.cpp).
Invoke either example through a bash shell script.

Using s_client
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Invoke the script using the IP address of the server as the first argument and
ECDHE or ECDHE_SHA256 as the second argument (ECDHE corresponding to ECDH ephemeral) when
connecting to a server using EC based credentials::

    > ./tlsSeClient.sh <server-IP-address> <ECDHE|ECDHE_256>

Invoke the script using the IP address of the server as the first argument and
RSA as the second argument when connecting to a server using RSA based credentials::

    > ./tlsSeClient.sh <server-IP-address> RSA

In case OpenSSL 1.1.1 is available on *both* Client (i.MX or Raspberry Pi) and Server side,
it's possible to request the usage of the TLS1.3 protocol (by default TLS1.2 is used). This is
achieved by setting the environment variable REQ_TLS to tls1_3::

    > REQ_TLS=tls1_3 ./tlsSeClient.sh <server-IP-address> ECDHE

Using tlsSe050Client.cpp
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

First compile the client program. Ensure all required SE050 specific libraries
and header files have been installed on the linux system. By default this example
links to static libraries::

    > cd demos/linux/tls_client/build
    > cmake ../.
    > cmake --build .

Invoke the script using the IP address of the server as argument::

    > ./tlsExtendedSeClient.sh <server-IP-address> <EC|RSA>

.. note::

    The environment variable REQ_TLS is not applicable to this example.
    In case OpenSSL 1.1.1 is available on the client, the actual TLS protocol version used will
    be negotiated to the highest version supported by both client and server.
    Otherwise TLS1.2 will be used.


.. _tls-client-using-a71ch:

TLS client example using A71CH
========================================================

Introduction
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The TLS client example can also be used in combination with an
A71CH secure element (in EC mode only).
The following steps are different::

- secure element preparation
- client program invocation

Secure Element preparation (client side)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Specify an additional option ``--subsystem a71ch``::

    > python3 provisionTlsClient.py --key_type ecc --subsystem a71ch

Invocation of client program
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Set the environment variable ``REQ_SE`` to ``a71ch`` when invoking the client program::

    > REQ_SE=a71ch ./tlsSeClient.sh 192.168.1.190 ECDHE
    > ... or ...
    > REQ_SE=a71ch ./tlsExtendedSeClient.sh 192.168.1.190 EC
